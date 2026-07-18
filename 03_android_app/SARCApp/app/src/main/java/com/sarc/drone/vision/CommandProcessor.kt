package com.sarc.drone.vision

import android.util.Log
import org.json.JSONObject

class CommandProcessor(
    private val trackingController: TrackingController,
    private val droneController: MavlinkController?,
    private val mqttManager: MqttManager?,
    private val onMissionAbort: () -> Unit,
    private val onChangeTarget: (personId: String) -> Unit,
    private val onFollowTarget: (enabled: Boolean) -> Unit,
    private val onSearchPattern: (pattern: String) -> Unit,
) {
    companion object {
        private const val TAG = "CommandProcessor"
    }

    fun processCommand(commandJson: JSONObject) {
        val cmdType = commandJson.optString("type")
        when (cmdType) {
            "CHANGE_TARGET" -> {
                val personId = commandJson.optString("person_id")
                Log.d(TAG, "Cambiando objetivo a persona: $personId")
                onChangeTarget(personId)
                ackCommand(commandJson.optString("id"), "OK")
            }

            "ABORT_MISSION" -> {
                Log.d(TAG, "Abortando mision")
                trackingController.reset()
                droneController?.disarm()
                onMissionAbort()
                ackCommand(commandJson.optString("id"), "ABORTED")
            }

            "SET_SEARCH_PATTERN" -> {
                val pattern = commandJson.optString("pattern", "lawnmower")
                onSearchPattern(pattern)
                ackCommand(commandJson.optString("id"), "PATTERN_SET")
            }

            "FOLLOW_TARGET" -> {
                val enabled = commandJson.optBoolean("enabled", true)
                onFollowTarget(enabled)
                ackCommand(commandJson.optString("id"), if (enabled) "FOLLOW_ENABLED" else "FOLLOW_DISABLED")
            }

            "MANUAL_CONTROL" -> {
                val vx = commandJson.optDouble("vx", 0.0).toFloat()
                val vy = commandJson.optDouble("vy", 0.0).toFloat()
                val vz = commandJson.optDouble("vz", 0.0).toFloat()
                val yaw = commandJson.optDouble("yaw", 0.0).toFloat()
                droneController?.setVelocityTarget(vx, vy, vz, yaw)
                ackCommand(commandJson.optString("id"), "MANUAL")
            }

            "REQUEST_TELEMETRY" -> {
                droneController?.requestTelemetry()
                ackCommand(commandJson.optString("id"), "TELEMETRY_REQUESTED")
            }

            else -> Log.w(TAG, "Comando desconocido: $cmdType")
        }
    }

    private fun ackCommand(cmdId: String, status: String) {
        val ack = JSONObject().apply {
            put("command_id", cmdId)
            put("status", status)
            put("timestamp", System.currentTimeMillis())
        }
        mqttManager?.publish(Config.TOPIC_ACK, ack)
        Log.d(TAG, "ACK enviado: $ack")
    }
}
