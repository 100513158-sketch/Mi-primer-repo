package com.sarc.drone.vision

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlin.math.hypot
import kotlin.random.Random

class DroneSimulator(
    private val onTelemetry: (lat: Double, lon: Double, alt: Double, heading: Float, speed: Float) -> Unit,
) {
    private var simulationJob: Job? = null
    private var currentLat = 40.7128
    private var currentLon = -74.0060
    private var currentAlt = 30.0
    private var currentHeading = 0f
    private var currentSpeed = 0f
    private var targetLat = currentLat
    private var targetLon = currentLon

    fun start() {
        if (simulationJob?.isActive == true) return

        simulationJob = CoroutineScope(Dispatchers.IO).launch {
            while (isActive) {
                val dx = targetLon - currentLon
                val dy = targetLat - currentLat
                val distance = hypot(dx, dy)
                if (distance > 0.00001) {
                    val step = 0.00001 * currentSpeed.coerceAtLeast(1.0f)
                    currentLon += dx / distance * step
                    currentLat += dy / distance * step
                }

                currentHeading = ((currentHeading + Random.nextInt(-5, 5)) % 360 + 360) % 360
                currentSpeed = (currentSpeed * 0.9f + Random.nextFloat() * 2f).coerceIn(0f, 15f)

                onTelemetry(currentLat, currentLon, currentAlt, currentHeading, currentSpeed)
                delay(100)
            }
        }
    }

    fun setTarget(lat: Double, lon: Double) {
        targetLat = lat
        targetLon = lon
    }

    fun setSpeed(speed: Float) {
        currentSpeed = speed
    }

    fun setAltitude(alt: Double) {
        currentAlt = alt
    }

    fun stop() {
        simulationJob?.cancel()
        simulationJob = null
    }
}
