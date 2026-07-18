package com.sarc.drone.vision

class BatteryManager(
    private val criticalLevel: Int,
    private val returnHomeLevel: Int,
    private val homePosition: Triple<Double, Double, Double>,
) {
    var currentBattery: Int = 100
        set(value) {
            field = value.coerceIn(0, 100)
            if (field <= returnHomeLevel) {
                onReturnToHome?.invoke(homePosition)
            }
            if (field <= criticalLevel) {
                onCriticalLanding?.invoke()
            }
        }

    var onReturnToHome: ((Triple<Double, Double, Double>) -> Unit)? = null
    var onCriticalLanding: (() -> Unit)? = null
}
