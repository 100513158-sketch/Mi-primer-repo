package com.sarc.drone.vision

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

enum class SearchPattern {
    LAWNMOWER,
    SPIRAL,
    WAYPOINT,
}

class SearchPlanner {
    companion object {
        fun generateWaypoints(
            pattern: SearchPattern,
            centerLat: Double,
            centerLon: Double,
            radius: Double,
            spacing: Double,
        ): List<Pair<Double, Double>> {
            return when (pattern) {
                SearchPattern.LAWNMOWER -> lawnmower(centerLat, centerLon, radius, spacing)
                SearchPattern.SPIRAL -> spiral(centerLat, centerLon, radius, spacing)
                SearchPattern.WAYPOINT -> emptyList()
            }
        }

        private fun lawnmower(
            centerLat: Double,
            centerLon: Double,
            radius: Double,
            spacing: Double,
        ): List<Pair<Double, Double>> {
            val points = mutableListOf<Pair<Double, Double>>()
            val delta = spacing / 111000.0
            val halfSide = radius / 111000.0
            val startLat = centerLat - halfSide
            val startLon = centerLon - halfSide
            val rows = ((2 * halfSide) / delta).toInt().coerceAtLeast(1)

            for (i in 0..rows) {
                val lat = startLat + i * delta
                val lonA = startLon
                val lonB = startLon + 2 * halfSide
                if (i % 2 == 0) {
                    points.add(Pair(lat, lonA))
                    points.add(Pair(lat, lonB))
                } else {
                    points.add(Pair(lat, lonB))
                    points.add(Pair(lat, lonA))
                }
            }
            return points
        }

        private fun spiral(
            centerLat: Double,
            centerLon: Double,
            radius: Double,
            spacing: Double,
        ): List<Pair<Double, Double>> {
            val points = mutableListOf<Pair<Double, Double>>()
            var r = spacing
            var angle = 0.0
            val deltaAngle = PI / 8.0

            while (r <= radius) {
                val lat = centerLat + (r / 111000.0) * cos(angle)
                val lon = centerLon + (r / 111000.0) * sin(angle)
                points.add(Pair(lat, lon))
                angle += deltaAngle
                r += spacing * deltaAngle / (2 * PI)
            }

            return points
        }
    }
}
