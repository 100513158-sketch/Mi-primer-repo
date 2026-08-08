param(
    [switch]$Release,
    [switch]$DeployToDevice
)

Write-Host "=== SARC-Drone Android App Build Script ===" -ForegroundColor Cyan
Set-Location "C:\SARC-Drone\03_android_app\SARCApp"

$assetDir = "app\src\main\assets"
if (-not (Test-Path $assetDir)) {
    New-Item -ItemType Directory -Path $assetDir | Out-Null
}

$candidateModels = @(
    "C:\SARC-Drone\02_models\exported\tflite\best_C2A\best_C2A.tflite",
    "C:\SARC-Drone\02_models\exported\tflite\best_detection\best_detection.tflite",
    "C:\SARC-Drone\02_models\exported\tflite\production_model.tflite"
)

$selectedModel = $null
foreach ($candidate in $candidateModels) {
    if (Test-Path $candidate) {
        $selectedModel = $candidate
        break
    }
}

if ($selectedModel) {
    $assetModel = Join-Path $assetDir "best_detection.tflite"
    Copy-Item $selectedModel $assetModel -Force
    Write-Host "Modelo TFLite preparado en assets desde: $selectedModel" -ForegroundColor Green
} else {
    Write-Host "Aviso: No se encontro modelo TFLite exportado. Se compila sin actualizar assets." -ForegroundColor Yellow
}

if ($Release) {
    Write-Host "Compilando version RELEASE..." -ForegroundColor Yellow
    .\gradlew assembleRelease
    $candidateApks = @(
        "app\build\outputs\apk\release\app-release.apk",
        "app\build\outputs\apk\release\app-release-unsigned.apk"
    )
} else {
    Write-Host "Compilando version DEBUG..." -ForegroundColor Yellow
    .\gradlew assembleDebug
    $candidateApks = @(
        "app\build\outputs\apk\debug\app-debug.apk"
    )
}

$apkPath = $null
foreach ($candidate in $candidateApks) {
    if (Test-Path $candidate) {
        $apkPath = $candidate
        break
    }
}

if ($apkPath) {
    Write-Host "APK generada exitosamente: $apkPath" -ForegroundColor Green
    Write-Host "Tamano: $([math]::Round((Get-Item $apkPath).Length / 1MB, 2)) MB" -ForegroundColor Green
} else {
    Write-Host "Error: No se pudo generar la APK" -ForegroundColor Red
    exit 1
}

if ($DeployToDevice) {
    Write-Host "Desplegando a dispositivo Android..." -ForegroundColor Yellow

    $devices = adb devices | Select-String -Pattern "device$" | ForEach-Object { $_.ToString().Split()[0] }
    if ($devices.Count -eq 0) {
        Write-Host "No se encontraron dispositivos Android conectados" -ForegroundColor Red
        exit 1
    }

    adb install -r $apkPath

    $modelPath = $selectedModel
    if (Test-Path $modelPath) {
        adb push $modelPath "/storage/emulated/0/Android/data/com.sarc.drone.vision/files/"
        Write-Host "Modelo copiado al dispositivo" -ForegroundColor Green
    }

    adb shell am start -n com.sarc.drone.vision/.MainActivity
    Write-Host "Aplicacion iniciada" -ForegroundColor Green
}

Write-Host "=== Build completado ===" -ForegroundColor Cyan
