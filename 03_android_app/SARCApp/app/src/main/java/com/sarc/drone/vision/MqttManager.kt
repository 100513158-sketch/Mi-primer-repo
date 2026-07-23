package com.sarc.drone.vision

import android.content.Context
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import org.eclipse.paho.client.mqttv3.IMqttActionListener
import org.eclipse.paho.client.mqttv3.IMqttDeliveryToken
import org.eclipse.paho.client.mqttv3.IMqttToken
import org.eclipse.paho.client.mqttv3.MqttAsyncClient
import org.eclipse.paho.client.mqttv3.MqttCallback
import org.eclipse.paho.client.mqttv3.MqttConnectOptions
import org.eclipse.paho.client.mqttv3.MqttException
import org.eclipse.paho.client.mqttv3.MqttMessage
import org.eclipse.paho.client.mqttv3.persist.MemoryPersistence
import org.json.JSONObject

class MqttManager(private val context: Context) {
    companion object {
        private const val TAG = "MqttManager"
        private const val RECONNECT_COOLDOWN_MS = 3000L
    }

    private var mqttClient: MqttAsyncClient? = null
    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    private var commandCallback: ((String, JSONObject) -> Unit)? = null
    @Volatile
    private var isConnected = false
    @Volatile
    private var isConnecting = false
    private var lastReconnectAttemptMs = 0L

    fun connectFromConfig(onConnected: () -> Unit = {}) {
        val brokerUri = if (Config.MQTT_TLS) {
            "ssl://${Config.MQTT_BROKER}:${Config.MQTT_PORT}"
        } else {
            "tcp://${Config.MQTT_BROKER}:${Config.MQTT_PORT}"
        }

        connect(
            brokerUri = brokerUri,
            clientId = Config.MQTT_CLIENT_ID,
            username = Config.MQTT_USERNAME,
            password = Config.MQTT_PASSWORD,
            onConnected = onConnected,
        )
    }

    fun connect(
        brokerUri: String,
        clientId: String,
        username: String,
        password: String,
        onConnected: () -> Unit,
    ) {
        if (isConnected || isConnecting) {
            return
        }

        isConnecting = true
        mqttClient = MqttAsyncClient(brokerUri, clientId, MemoryPersistence())
        mqttClient?.setCallback(object : MqttCallback {
            override fun connectionLost(cause: Throwable?) {
                isConnected = false
                isConnecting = false
                Log.e(TAG, "Conexion MQTT perdida", cause)
                scope.launch {
                    delay(5000)
                    requestReconnect()
                }
            }

            override fun messageArrived(topic: String, message: MqttMessage) {
                val payload = String(message.payload)
                Log.d(TAG, "Mensaje recibido en $topic: $payload")
                try {
                    val json = JSONObject(payload)
                    commandCallback?.invoke(topic, json)
                } catch (e: Exception) {
                    Log.e(TAG, "Error parseando comando", e)
                }
            }

            override fun deliveryComplete(token: IMqttDeliveryToken?) {
            }
        })

        val options = MqttConnectOptions().apply {
            userName = username
            this.password = password.toCharArray()
            isCleanSession = true
            connectionTimeout = 30
            keepAliveInterval = Config.MQTT_KEEPALIVE
            isAutomaticReconnect = true
        }

        try {
            mqttClient?.connect(options, null, object : IMqttActionListener {
                override fun onSuccess(asyncActionToken: IMqttToken?) {
                    isConnected = true
                    isConnecting = false
                    Log.d(TAG, "MQTT conectado")
                    subscribeToCommands()
                    onConnected()
                }

                override fun onFailure(asyncActionToken: IMqttToken?, exception: Throwable?) {
                    isConnected = false
                    isConnecting = false
                    Log.e(TAG, "Error MQTT: ${exception?.message}")
                }
            })
        } catch (e: MqttException) {
            isConnected = false
            isConnecting = false
            Log.e(TAG, "Excepcion MQTT", e)
        }
    }

    private fun requestReconnect() {
        val now = System.currentTimeMillis()
        if (now - lastReconnectAttemptMs < RECONNECT_COOLDOWN_MS) {
            return
        }
        lastReconnectAttemptMs = now
        connectFromConfig()
    }

    private fun subscribeToCommands() {
        val topic = "${Config.TOPIC_COMMANDS}/#"
        mqttClient?.subscribe(topic, 1, null, object : IMqttActionListener {
            override fun onSuccess(asyncActionToken: IMqttToken?) {
                Log.d(TAG, "Suscrito a $topic")
            }

            override fun onFailure(asyncActionToken: IMqttToken?, exception: Throwable?) {
                Log.e(TAG, "Fallo suscripcion a comandos", exception)
            }
        })
    }

    fun publish(topic: String, payload: JSONObject) {
        if (mqttClient == null || !isConnected) {
            Log.w(TAG, "Client not connected. Skipping publish and requesting reconnect")
            requestReconnect()
            return
        }

        val message = MqttMessage(payload.toString().toByteArray()).apply {
            qos = 1
        }
        mqttClient?.publish(topic, message, null, object : IMqttActionListener {
            override fun onSuccess(asyncActionToken: IMqttToken?) {
                Log.d(TAG, "Publicado en $topic")
            }

            override fun onFailure(asyncActionToken: IMqttToken?, exception: Throwable?) {
                isConnected = false
                Log.e(TAG, "Fallo publicacion", exception)
                requestReconnect()
            }
        })
    }

    fun setCommandListener(listener: (topic: String, command: JSONObject) -> Unit) {
        commandCallback = listener
    }

    fun disconnect() {
        try {
            isConnected = false
            isConnecting = false
            mqttClient?.disconnect()
            mqttClient?.close()
            mqttClient = null
        } catch (_: Exception) {
        }
    }
}
