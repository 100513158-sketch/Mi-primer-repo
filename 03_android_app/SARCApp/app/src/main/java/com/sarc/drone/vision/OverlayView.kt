package com.sarc.drone.vision

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View

class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 5f
    }

    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.GREEN
        textSize = 40f
        style = Paint.Style.FILL
    }

    private val trackingPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.YELLOW
        style = Paint.Style.STROKE
        strokeWidth = 8f
    }

    private val targetPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.RED
        style = Paint.Style.FILL
    }

    private var detections: List<Detection> = emptyList()
    private var isTracking = false
    private var trackX = 0f
    private var trackY = 0f

    fun updateDetections(newDetections: List<Detection>) {
        detections = newDetections
        invalidate()
    }

    fun startTracking(x: Float, y: Float) {
        isTracking = true
        trackX = x
        trackY = y
        invalidate()
    }

    fun updateTracking(x: Float, y: Float) {
        isTracking = true
        trackX = x
        trackY = y
        invalidate()
    }

    fun clearTracking() {
        isTracking = false
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        for (detection in detections) {
            canvas.drawRect(detection.boundingBox, paint)
            val label = "${detection.className} ${(detection.confidence * 100).toInt()}%"
            canvas.drawText(label, detection.boundingBox.left, detection.boundingBox.top - 10, textPaint)
        }

        if (isTracking) {
            canvas.drawCircle(trackX, trackY, 30f, targetPaint)
            canvas.drawCircle(trackX, trackY, 50f, trackingPaint)
            canvas.drawLine(trackX - 80, trackY, trackX - 30, trackY, trackingPaint)
            canvas.drawLine(trackX + 30, trackY, trackX + 80, trackY, trackingPaint)
            canvas.drawLine(trackX, trackY - 80, trackX, trackY - 30, trackingPaint)
            canvas.drawLine(trackX, trackY + 30, trackX, trackY + 80, trackingPaint)
        }
    }
}
