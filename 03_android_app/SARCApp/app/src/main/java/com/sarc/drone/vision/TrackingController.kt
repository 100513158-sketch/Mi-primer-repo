package com.sarc.drone.vision

import android.util.Log
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt

class TrackingController {

    companion object {
        private const val TAG = "TrackingController"
    }

    private var integralX = 0f
    private var integralY = 0f
    private var lastErrorX = 0f
    private var lastErrorY = 0f
    private var lastTimestamp = System.currentTimeMillis()

    private var currentDistance = Config.TARGET_DISTANCE
    private var targetLocked = false

    data class ControlCommand(
        val velocityX: Float,
        val velocityY: Float,
        val velocityZ: Float,
        val yawRate: Float,
        val approachDistance: Float,
        val maintainTarget: Boolean
    )

    fun computeControl(
        offsetX: Float,
        offsetY: Float,
        targetWidth: Float,
        targetHeight: Float,
        frameWidth: Float,
        frameHeight: Float
    ): ControlCommand {

        val currentTime = System.currentTimeMillis()
        val dt = ((currentTime - lastTimestamp).coerceAtLeast(1L)) / 1000f
        lastTimestamp = currentTime

        val targetArea = targetWidth * targetHeight
        val frameArea = frameWidth * frameHeight
        val areaRatio = targetArea / max(frameArea, 1f)

        val estimatedDistance = Config.TARGET_DISTANCE * sqrt(0.05f / max(areaRatio, 0.01f))
        currentDistance = estimatedDistance.coerceIn(Config.SAFE_DISTANCE_MIN, Config.SAFE_DISTANCE_MAX)

        val errorX = -offsetX
        val errorY = -offsetY

        integralX = (integralX + errorX * dt).coerceIn(-1f, 1f)
        integralY = (integralY + errorY * dt).coerceIn(-1f, 1f)

        val derivativeX = (errorX - lastErrorX) / dt
        val derivativeY = (errorY - lastErrorY) / dt

        lastErrorX = errorX
        lastErrorY = errorY

        val velocityX = (Config.KP * errorX + Config.KI * integralX + Config.KD * derivativeX).coerceIn(-3f, 3f)
        val velocityY = (Config.KP * errorY + Config.KI * integralY + Config.KD * derivativeY).coerceIn(-3f, 3f)

        val distanceError = (currentDistance - Config.TARGET_DISTANCE) / Config.TARGET_DISTANCE
        val velocityZ = (distanceError * 1.5f).coerceIn(-2f, 2f)
        val yawRate = (errorX * 1.2f).coerceIn(-0.5f, 0.5f)

        val isCentered = abs(errorX) < Config.CENTER_THRESHOLD && abs(errorY) < Config.CENTER_THRESHOLD
        if (isCentered && !targetLocked) {
            targetLocked = true
            Log.d(TAG, "Target locked and centered")
        }

        return ControlCommand(
            velocityX = velocityX,
            velocityY = velocityY,
            velocityZ = velocityZ,
            yawRate = yawRate,
            approachDistance = Config.TARGET_DISTANCE.toFloat(),
            maintainTarget = targetLocked
        )
    }

    fun reset() {
        integralX = 0f
        integralY = 0f
        lastErrorX = 0f
        lastErrorY = 0f
        targetLocked = false
        currentDistance = Config.TARGET_DISTANCE
        Log.d(TAG, "Tracking controller reset")
    }

    fun isTargetLocked(): Boolean = targetLocked
}
