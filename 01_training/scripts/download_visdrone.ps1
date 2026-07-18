# download_visdrone.ps1
# Lee URL y lista de archivos desde config.yaml para evitar valores hardcodeados.

$projectRoot = (Resolve-Path "$PSScriptRoot\..\..").Path
$configPath = Join-Path $projectRoot "config.yaml"
$visdroneDir = Join-Path $projectRoot "00_datasets\raw\visdrone"

if (!(Test-Path $configPath)) {
    throw "No se encontro config.yaml en $configPath"
}

if (!(Test-Path $visdroneDir)) {
    New-Item -ItemType Directory -Path $visdroneDir | Out-Null
}

$json = python -c "import yaml, json; c=yaml.safe_load(open(r'$configPath','r',encoding='utf-8')); print(json.dumps(c['datasets']['visdrone']))"
if (-not $json) {
    throw "No se pudo leer datasets.visdrone desde config.yaml"
}

$cfg = $json | ConvertFrom-Json
$baseUrl = $cfg.url

Push-Location $visdroneDir
try {
    foreach ($file in $cfg.files) {
        Write-Host "Descargando $file..."
        Invoke-WebRequest -Uri "$baseUrl/$file" -OutFile $file
        Expand-Archive -Path $file -DestinationPath "." -Force
        Remove-Item $file -Force
    }
}
finally {
    Pop-Location
}

Write-Host "VisDrone descargado correctamente en $visdroneDir"