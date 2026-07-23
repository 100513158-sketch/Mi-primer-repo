# Reglas locales de ProGuard/R8 para SARCApp.

# Keep TensorFlow Lite classes used by runtime delegate loading.
-keep class org.tensorflow.lite.** { *; }
-keep class org.tensorflow.lite.gpu.** { *; }
-dontwarn org.tensorflow.lite.**
