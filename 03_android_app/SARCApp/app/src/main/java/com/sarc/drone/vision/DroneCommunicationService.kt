package com.sarc.drone.vision

import android.app.Service
import android.content.Intent
import android.os.IBinder
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.InetAddress

class DroneCommunicationService : Service() {

    companion object {
        private const val TAG = "DroneCommService"

        private var udpSocket: DatagramSocket? = null
        private var isConnected = false

        @JvmStatic
        fun sendCommand(command: String) {
            if (!isConnected || udpSocket == null) {
                Log.e(TAG, "No hay conexion UDP activa")
                return
            }
            try {
                val data = command.toByteArray()
                val address = InetAddress.getByName(Config.DRONE_IP)
                val packet = DatagramPacket(data, data.size, address, Config.DRONE_UDP_PORT)
                udpSocket?.send(packet)
                Log.d(TAG, "Comando enviado: $command")
            } catch (e: Exception) {
                Log.e(TAG, "Error enviando comando", e)
            }
        }
    }

    private val serviceScope = CoroutineScope(Dispatchers.IO + SupervisorJob())

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        setupUdpConnection()
        startListening()
    }

    private fun setupUdpConnection() {
        try {
            udpSocket = DatagramSocket(Config.LOCAL_UDP_PORT)
            udpSocket?.soTimeout = 1000
            isConnected = true
            Log.d(TAG, "Conexion UDP establecida en puerto ${Config.LOCAL_UDP_PORT}")
        } catch (e: Exception) {
            Log.e(TAG, "Error estableciendo conexion UDP", e)
            isConnected = false
        }
    }

    private fun startListening() {
        serviceScope.launch {
            while (isConnected) {
                try {
                    val buffer = ByteArray(1024)
                    val packet = DatagramPacket(buffer, buffer.size)
                    udpSocket?.receive(packet)
                    val message = String(packet.data, 0, packet.length)
                    Log.d(TAG, "Mensaje recibido del dron: $message")
                } catch (e: Exception) {
                    if (isConnected) {
                        Log.e(TAG, "Error recibiendo mensaje", e)
                    }
                    delay(100)
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        isConnected = false
        udpSocket?.close()
        serviceScope.cancel()
    }
}
