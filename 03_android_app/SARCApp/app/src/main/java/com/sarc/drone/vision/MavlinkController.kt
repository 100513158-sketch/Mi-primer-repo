package com.sarc.drone.vision

import android.util.Log
import io.dronefleet.mavlink.MavlinkConnection
import io.dronefleet.mavlink.common.CommandLong
import io.dronefleet.mavlink.common.MavCmd
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.net.Socket

class MavlinkController(private val droneIp: String, private val dronePort: Int) {

    companion object {
        private const val TAG = "MavlinkController"
        private const val TARGET_SYSTEM = 1
        private const val TARGET_COMPONENT = 1
    }

    private var socket: Socket? = null
    private var connection: MavlinkConnection? = null

    fun connect(): Boolean {
        return try {
            socket = Socket(droneIp, dronePort)
            connection = MavlinkConnection.create(
                BufferedInputStream(socket!!.getInputStream()),
                BufferedOutputStream(socket!!.getOutputStream())
            )
            Log.d(TAG, "Conectado al dron en $droneIp:$dronePort")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Error conectando al dron", e)
            false
        }
    }

    fun arm() {
        sendCommandLong(MavCmd.MAV_CMD_COMPONENT_ARM_DISARM, 1.0f)
    }

    fun disarm() {
        sendCommandLong(MavCmd.MAV_CMD_COMPONENT_ARM_DISARM, 0.0f)
    }

    fun takeoff(altitude: Float = 30.0f) {
        sendCommandLong(MavCmd.MAV_CMD_NAV_TAKEOFF, param7 = altitude)
    }

    fun land() {
        sendCommandLong(MavCmd.MAV_CMD_NAV_LAND)
    }

    fun setVelocityTarget(velocityX: Float, velocityY: Float, velocityZ: Float, yawRate: Float) {
        // Placeholder MAVLink velocity command encoder.
        // In production, map this to SET_POSITION_TARGET_* messages.
        Log.d(TAG, "Velocity target vx=$velocityX vy=$velocityY vz=$velocityZ yawRate=$yawRate")
    }

    fun requestTelemetry() {
        // Placeholder request to simplify remote command flow wiring.
        Log.d(TAG, "Telemetry requested")
    }

    private fun sendCommandLong(
        command: MavCmd,
        param1: Float = 0.0f,
        param2: Float = 0.0f,
        param3: Float = 0.0f,
        param4: Float = 0.0f,
        param5: Float = 0.0f,
        param6: Float = 0.0f,
        param7: Float = 0.0f
    ) {
        val msg = CommandLong.builder()
            .targetSystem(TARGET_SYSTEM)
            .targetComponent(TARGET_COMPONENT)
            .command(command)
            .confirmation(0)
            .param1(param1)
            .param2(param2)
            .param3(param3)
            .param4(param4)
            .param5(param5)
            .param6(param6)
            .param7(param7)
            .build()

        connection?.send2(255, 1, msg)
        Log.d(TAG, "Comando enviado: $command")
    }

    fun disconnect() {
        try {
            socket?.close()
        } catch (_: Exception) {
        }
        socket = null
        connection = null
        Log.d(TAG, "Desconectado del dron")
    }
}
