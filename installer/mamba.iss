; ============================================================================
;  Inno Setup script for Mamba Terminal by Quincy Gininda
;
;  Produces a proper Windows installer (MambaTerminalSetup.exe) that installs
;  the two apps, adds Start Menu + optional desktop shortcuts, and registers an
;  uninstaller. Compiled by Inno Setup's ISCC, either locally or in the cloud
;  (see appveyor.yml). It bundles the PyInstaller output in ..\dist, so build
;  the exes first:
;      pyinstaller --clean --noconfirm packaging\mamba-terminal.spec
;      pyinstaller --clean --noconfirm packaging\mamba-web.spec
;      iscc installer\mamba.iss
; ============================================================================

#define AppName        "Mamba Terminal"
#define AppPublisher   "Quincy Gininda"
#define AppVersion     "0.1.0"
#define AppExeWeb      "mamba-web.exe"
#define AppExeTerminal "mamba-terminal.exe"

[Setup]
AppId={{9F5B4E2A-3C7D-4B1E-9A6F-1A2B3C4D5E6F}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppVerName={#AppName} {#AppVersion}
DefaultDirName={autopf}\Mamba Terminal
DefaultGroupName=Mamba Terminal
DisableProgramGroupPage=yes
OutputDir=..\installer_output
OutputBaseFilename=MambaTerminalSetup
Compression=lzma2/max
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern
UninstallDisplayIcon={app}\{#AppExeWeb}
UninstallDisplayName={#AppName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"

[Files]
Source: "..\dist\{#AppExeWeb}";      DestDir: "{app}"; Flags: ignoreversion
Source: "..\dist\{#AppExeTerminal}"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\BUILD_WINDOWS.md";       DestDir: "{app}"; Flags: ignoreversion isreadme

[Icons]
Name: "{group}\Mamba Terminal (Web HUD)"; Filename: "{app}\{#AppExeWeb}"
Name: "{group}\Mamba Terminal (Console)"; Filename: "{app}\{#AppExeTerminal}"
Name: "{group}\Uninstall Mamba Terminal"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Mamba Terminal";     Filename: "{app}\{#AppExeWeb}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeWeb}"; Description: "Launch Mamba Terminal now"; Flags: nowait postinstall skipifsilent
