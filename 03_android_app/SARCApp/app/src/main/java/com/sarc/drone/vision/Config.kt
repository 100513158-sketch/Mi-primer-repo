package com.sarc.drone.vision

object Config {
    const val MQTT_CLIENT_ID = "sarc_drone_001"
    const val DRONE_ID = MQTT_CLIENT_ID
    const val MODEL_FILENAME = "best_detection.tflite"
    const val INPUT_SIZE = 640
    const val CONFIDENCE_THRESHOLD = 0.45f
    const val NMS_THRESHOLD = 0.5f

    const val KP = 0.8f
    const val KI = 0.1f
    const val KD = 0.3f
    const val SAFE_DISTANCE_MIN = 5.0f
    const val SAFE_DISTANCE_MAX = 30.0f
    const val TARGET_DISTANCE = 15.0f
    const val CENTER_THRESHOLD = 0.05f

    const val DRONE_UDP_PORT = 14550
    const val LOCAL_UDP_PORT = 14551
    const val DRONE_IP = "192.168.1.1"

    const val MQTT_BROKER = "192.168.1.134"
    const val MQTT_PORT = 1883
    const val MQTT_TLS = false
    const val MQTT_USERNAME = "drone_client"
    const val MQTT_PASSWORD = "change_me_drone_client"
    const val MQTT_KEEPALIVE = 60

    const val TOPIC_TELEMETRY = "sarc/drone/telemetry"
    const val TOPIC_DETECTIONS = "sarc/drone/detections"
    const val TOPIC_TRACKING = "sarc/drone/tracking"
    const val TOPIC_COMMANDS = "sarc/commands/drone"
    const val TOPIC_ACK = "sarc/drone/ack"

    const val SIMULATION_ENABLED = true
    const val BATTERY_CRITICAL_LEVEL = 15
    const val BATTERY_RTH_LEVEL = 25
    val HOME_POSITION = Triple(0.0, 0.0, 30.0)

    const val COLLISION_AVOIDANCE_ENABLED = true
    const val SENSOR_RANGE_M = 10.0f
    const val SAFETY_MARGIN_M = 2.0f
}
