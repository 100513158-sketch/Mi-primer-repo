package com.sarc.drone.vision

import kotlin.math.sqrt

data class Obstacle(
    val x: Double,
    val y: Double,
)

class ObstacleAvoidance(
    private val sensorRange: Double,
    private val safetyMargin: Double,
) {
    fun calculateAvoidanceVector(
        obstacles: List<Obstacle>,
        currentPos: Pair<Double, Double>,
        currentHeading: Float,
    ): Pair<Float, Float> {
        var avoidanceX = 0f
        var avoidanceY = 0f

        for (obs in obstacles) {
            val dx = obs.x - currentPos.first
            val dy = obs.y - currentPos.second
            val distance = sqrt(dx * dx + dy * dy).coerceAtLeast(1e-6)
            if (distance < sensorRange + safetyMargin) {
                val factor = (1.0 - (distance / (sensorRange + safetyMargin))).toFloat().coerceIn(0f, 1f)
                avoidanceX += (dx / distance).toFloat() * factor
                avoidanceY += (dy / distance).toFloat() * factor
            }
        }

        return Pair(-avoidanceX, -avoidanceY)
    }
}
