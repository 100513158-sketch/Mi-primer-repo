package com.sarc.drone.vision

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.RectF
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.android.material.switchmaterial.SwitchMaterial
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.Executors

class VisionActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "VisionActivity"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(
            Manifest.permission.CAMERA,
            Manifest.permission.INTERNET
        )
        private const val LABELS_FILENAME = "labels.txt"
    }

    private lateinit var previewView: PreviewView
    private lateinit var overlayView: OverlayView
    private lateinit var fpsTextView: TextView
    private lateinit var detectionCountText: TextView
    private lateinit var toggleSwitch: SwitchMaterial
    private lateinit var manualControlBtn: Button

    private var tfliteInterpreter: Interpreter? = null
    private var labels: List<String> = emptyList()
    private var gpuDelegate: GpuDelegate? = null

    private var mqttManager: MqttManager? = null
    private var commandProcessor: CommandProcessor? = null
    private var droneSimulator: DroneSimulator? = null
    private var batteryManager: BatteryManager? = null
    private var mavlinkController: MavlinkController? = null
    private val trackingController = TrackingController()

    private val analysisExecutor = Executors.newSingleThreadExecutor()

    private var fpsUpdateTime = System.currentTimeMillis()
    private var currentFps = 0f
    private var trackingEnabled = true
    private var currentDroneLat = 0.0
    private var currentDroneLon = 0.0
    private var currentDroneAlt = 0.0
    private var currentSearchPattern = "lawnmower"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_vision)

        initializeViews()
        setupIoTAndAutonomy()
        loadModel()
        requestPermissions()

        manualControlBtn.setOnClickListener {
            DroneCommunicationService.sendCommand("MANUAL_CONTROL_REQUEST")
        }
    }

    private fun setupIoTAndAutonomy() {
        mqttManager = MqttManager(this)
        mavlinkController = MavlinkController(Config.DRONE_IP, Config.DRONE_UDP_PORT)

        commandProcessor = CommandProcessor(
            trackingController = trackingController,
            droneController = mavlinkController,
            mqttManager = mqttManager,
            onMissionAbort = {
                runOnUiThread {
                    trackingEnabled = false
                    overlayView.clearTracking()
                    Toast.makeText(this, "Mision abortada", Toast.LENGTH_SHORT).show()
                }
            },
            onChangeTarget = { personId ->
                Log.d(TAG, "Nuevo objetivo remoto: $personId")
            },
            onFollowTarget = { enabled ->
                trackingEnabled = enabled
                if (enabled) {
                    overlayView.clearTracking()
                    Log.d(TAG, "Modo follow activado desde backend")
                } else {
                    overlayView.clearTracking()
                    Log.d(TAG, "Modo follow desactivado desde backend")
                }
            },
            onSearchPattern = { pattern ->
                currentSearchPattern = pattern
                Log.d(TAG, "Patron de busqueda remoto: $pattern")
            },
        )

        mqttManager?.setCommandListener { _, command ->
            commandProcessor?.processCommand(command)
        }
        mqttManager?.connectFromConfig()

        batteryManager = BatteryManager(
            criticalLevel = Config.BATTERY_CRITICAL_LEVEL,
            returnHomeLevel = Config.BATTERY_RTH_LEVEL,
            homePosition = Config.HOME_POSITION,
        ).apply {
            onReturnToHome = { home ->
                Log.w(TAG, "RTH activado hacia $home")
            }
            onCriticalLanding = {
                Log.e(TAG, "Bateria critica, aterrizaje de emergencia")
                mavlinkController?.land()
            }
        }

        if (Config.SIMULATION_ENABLED) {
            droneSimulator = DroneSimulator { lat, lon, alt, _, _ ->
                currentDroneLat = lat
                currentDroneLon = lon
                currentDroneAlt = alt
                publishTelemetry()
            }
            droneSimulator?.start()
        }
    }

    private fun initializeViews() {
        previewView = findViewById(R.id.previewView)
        overlayView = findViewById(R.id.overlayView)
        fpsTextView = findViewById(R.id.fpsTextView)
        detectionCountText = findViewById(R.id.detectionCountText)
        toggleSwitch = findViewById(R.id.toggleSwitch)
        manualControlBtn = findViewById(R.id.manualControlBtn)

        toggleSwitch.setOnCheckedChangeListener { _, isChecked ->
            trackingEnabled = isChecked
            if (!isChecked) {
                overlayView.clearTracking()
                DroneCommunicationService.sendCommand("STOP_TRACKING")
            }
        }
    }

    private fun loadModel() {
        try {
            val modelBuffer = loadModelFile(Config.MODEL_FILENAME)
            val options = Interpreter.Options().apply {
                setNumThreads(4)
                // Some devices or packaging combinations may fail loading GPU classes at runtime.
                // Fallback to CPU instead of crashing the app.
                try {
                    gpuDelegate = GpuDelegate()
                    addDelegate(gpuDelegate)
                    Log.d(TAG, "GPU delegate enabled")
                } catch (t: Throwable) {
                    gpuDelegate = null
                    Log.w(TAG, "GPU delegate unavailable, using CPU", t)
                }
            }
            tfliteInterpreter = Interpreter(modelBuffer, options)
            labels = FileUtil.loadLabels(this, LABELS_FILENAME)
            Toast.makeText(this, "Modelo de vision cargado", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Log.e(TAG, "Error cargando modelo", e)
            Toast.makeText(this, "Error cargando modelo: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    private fun loadModelFile(modelFilename: String): MappedByteBuffer {
        val assetFileDescriptor = assets.openFd(modelFilename)
        val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            val imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            imageAnalysis.setAnalyzer(analysisExecutor) { imageProxy ->
                processImage(imageProxy)
            }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis)
            } catch (e: Exception) {
                Log.e(TAG, "Error starting camera", e)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    @SuppressLint("UnsafeOptInUsageError")
    private fun processImage(imageProxy: ImageProxy) {
        val startTime = System.nanoTime()

        val detections = try {
            val bitmap = imageProxyToBitmap(imageProxy)
            val resizedBitmap = Bitmap.createScaledBitmap(bitmap, Config.INPUT_SIZE, Config.INPUT_SIZE, true)
            val inputBuffer = preprocessImage(resizedBitmap)
            val rawOutput = runInference(inputBuffer)
            parseDetections(rawOutput, imageProxy.width.toFloat(), imageProxy.height.toFloat())
        } catch (e: Exception) {
            Log.e(TAG, "Error procesando imagen", e)
            emptyList()
        } finally {
            imageProxy.close()
        }

        runOnUiThread {
            updateFps(startTime)
            detectionCountText.text = "Detecciones: ${detections.size}"
            overlayView.updateDetections(detections)

            if (trackingEnabled && detections.isNotEmpty()) {
                val first = detections.first()
                val centerX = first.boundingBox.centerX()
                val centerY = first.boundingBox.centerY()
                overlayView.updateTracking(centerX, centerY)
                DroneCommunicationService.sendCommand("TRACK:$centerX:$centerY")
                publishTracking(first, centerX, centerY)
                publishDetection(first)
            }
        }
    }

    private fun publishTelemetry() {
        val payload = JSONObject().apply {
            put("drone_id", Config.DRONE_ID)
            put("timestamp", System.currentTimeMillis())
            put("lat", currentDroneLat)
            put("lon", currentDroneLon)
            put("alt", currentDroneAlt)
            put("search_pattern", currentSearchPattern)
            put("fps", currentFps)
            put("source", "android_vision_app")
        }
        mqttManager?.publish(Config.TOPIC_TELEMETRY, payload)
    }

    private fun publishDetection(detection: Detection) {
        val json = JSONObject().apply {
            put("drone_id", Config.DRONE_ID)
            put("timestamp", System.currentTimeMillis())
            put("class", detection.className)
            put("confidence", detection.confidence)
            put("bbox", JSONObject().apply {
                put("x", detection.boundingBox.centerX())
                put("y", detection.boundingBox.centerY())
                put("w", detection.boundingBox.width())
                put("h", detection.boundingBox.height())
            })
            put("drone_position", JSONObject().apply {
                put("lat", currentDroneLat)
                put("lon", currentDroneLon)
                put("alt", currentDroneAlt)
            })
            put("fps", currentFps)
            put("source", "android_vision_app")
        }
        mqttManager?.publish(Config.TOPIC_DETECTIONS, json)
    }

    private fun publishTracking(detection: Detection, targetCenterX: Float, targetCenterY: Float) {
        val command = "TRACK:$targetCenterX:$targetCenterY"
        val json = JSONObject().apply {
            put("drone_id", Config.DRONE_ID)
            put("timestamp", System.currentTimeMillis())
            put("class", detection.className)
            put("confidence", detection.confidence)
            put("target", JSONObject().apply {
                put("x", targetCenterX)
                put("y", targetCenterY)
            })
            put("bbox", JSONObject().apply {
                put("x", detection.boundingBox.centerX())
                put("y", detection.boundingBox.centerY())
                put("w", detection.boundingBox.width())
                put("h", detection.boundingBox.height())
            })
            put("drone_position", JSONObject().apply {
                put("lat", currentDroneLat)
                put("lon", currentDroneLon)
                put("alt", currentDroneAlt)
            })
            put("fps", currentFps)
            put("tracking_enabled", trackingEnabled)
            put("command", command)
            put("source", "android_vision_app")
        }
        mqttManager?.publish(Config.TOPIC_TRACKING, json)
    }

    private fun parseDetections(rawOutput: FloatArray, imageWidth: Float, imageHeight: Float): List<Detection> {
        val stride = 6
        val detections = mutableListOf<Detection>()

        var index = 0
        while (index + stride - 1 < rawOutput.size) {
            val centerX = rawOutput[index]
            val centerY = rawOutput[index + 1]
            val width = rawOutput[index + 2]
            val height = rawOutput[index + 3]
            val confidence = rawOutput[index + 4]
            val classId = rawOutput[index + 5].toInt()

            if (confidence >= Config.CONFIDENCE_THRESHOLD) {
                val normalized = centerX in 0f..1.5f && centerY in 0f..1.5f && width in 0f..1.5f && height in 0f..1.5f
                val cx = if (normalized) centerX * imageWidth else centerX
                val cy = if (normalized) centerY * imageHeight else centerY
                val w = if (normalized) width * imageWidth else width
                val h = if (normalized) height * imageHeight else height

                val left = (cx - (w / 2f)).coerceIn(0f, imageWidth)
                val top = (cy - (h / 2f)).coerceIn(0f, imageHeight)
                val right = (cx + (w / 2f)).coerceIn(0f, imageWidth)
                val bottom = (cy + (h / 2f)).coerceIn(0f, imageHeight)
                val label = labels.getOrNull(classId) ?: "class_$classId"

                detections.add(
                    Detection(
                        classId = classId,
                        className = label,
                        confidence = confidence,
                        boundingBox = RectF(left, top, right, bottom)
                    )
                )
            }

            index += stride
        }

        return detections.sortedByDescending { it.confidence }.take(10)
    }

    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap {
        val nv21 = yuv420888ToNv21(imageProxy)
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, imageProxy.width, imageProxy.height, null)
        val outputStream = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, imageProxy.width, imageProxy.height), 90, outputStream)
        val bytes = outputStream.toByteArray()
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    }

    private fun yuv420888ToNv21(imageProxy: ImageProxy): ByteArray {
        val width = imageProxy.width
        val height = imageProxy.height
        val yPlane = imageProxy.planes[0]
        val uPlane = imageProxy.planes[1]
        val vPlane = imageProxy.planes[2]

        val nv21 = ByteArray(width * height * 3 / 2)
        var index = 0

        for (row in 0 until height) {
            val yRowStart = row * yPlane.rowStride
            for (col in 0 until width) {
                nv21[index++] = yPlane.buffer.get(yRowStart + col * yPlane.pixelStride)
            }
        }

        for (row in 0 until height / 2) {
            val uvRowStart = row * uPlane.rowStride
            for (col in 0 until width / 2) {
                val uvIndex = uvRowStart + col * uPlane.pixelStride
                nv21[index++] = vPlane.buffer.get(uvIndex)
                nv21[index++] = uPlane.buffer.get(uvIndex)
            }
        }

        return nv21
    }

    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        val inputBuffer = ByteBuffer.allocateDirect(1 * Config.INPUT_SIZE * Config.INPUT_SIZE * 3 * 4)
        inputBuffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(Config.INPUT_SIZE * Config.INPUT_SIZE)
        bitmap.getPixels(pixels, 0, Config.INPUT_SIZE, 0, 0, Config.INPUT_SIZE, Config.INPUT_SIZE)

        for (pixel in pixels) {
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f
            inputBuffer.putFloat(r)
            inputBuffer.putFloat(g)
            inputBuffer.putFloat(b)
        }

        return inputBuffer
    }

    private fun runInference(inputBuffer: ByteBuffer): FloatArray {
        val outputShape = intArrayOf(1, 8400, 6)
        val outputBuffer = TensorBuffer.createFixedSize(outputShape, DataType.FLOAT32)
        tfliteInterpreter?.run(inputBuffer, outputBuffer.buffer)
        return outputBuffer.floatArray
    }

    private fun updateFps(startTime: Long) {
        val elapsedMs = (System.nanoTime() - startTime) / 1_000_000.0
        val fpsNow = 1000.0 / elapsedMs
        currentFps = (currentFps * 0.9 + fpsNow * 0.1).toFloat()

        if (System.currentTimeMillis() - fpsUpdateTime > 1000) {
            fpsTextView.text = String.format("FPS: %.1f", currentFps)
            fpsUpdateTime = System.currentTimeMillis()
        }
    }

    private fun requestPermissions() {
        if (REQUIRED_PERMISSIONS.all {
                ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
            }) {
            startCamera()
            return
        }
        ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS && grantResults.all { it == PackageManager.PERMISSION_GRANTED }) {
            startCamera()
        } else {
            Toast.makeText(this, "Permisos no concedidos", Toast.LENGTH_LONG).show()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        droneSimulator?.stop()
        mqttManager?.disconnect()
        mavlinkController?.disconnect()
        tfliteInterpreter?.close()
        gpuDelegate?.close()
        analysisExecutor.shutdown()
    }
}

data class Detection(
    val classId: Int,
    val className: String,
    val confidence: Float,
    val boundingBox: RectF
)
