[Setup]
AppName=Buki
AppVersion=1.0
AppPublisher=Buki
AppPublisherURL=https://github.com
DefaultDirName={localappdata}\Buki
DefaultGroupName=Buki
OutputDir=C:\Users\Bryan\buki\output
OutputBaseFilename=Buki-Setup
PrivilegesRequired=lowest
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
DisableProgramGroupPage=yes
UninstallDisplayName=Buki
UninstallDisplayIcon={app}\buki.py

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
Source: "C:\Users\Bryan\buki\buki.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\Bryan\buki\install_deps.ps1"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\Buki"; Filename: "{app}\launch.vbs"; WorkingDir: "{app}"
Name: "{userdesktop}\Buki"; Filename: "{app}\launch.vbs"; WorkingDir: "{app}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a Desktop shortcut"; GroupDescription: "Additional icons:"

[Run]
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\install_deps.ps1"""; StatusMsg: "Installing Python and dependencies (this may take a few minutes)..."; Flags: runhidden waituntilterminated

[Code]
procedure CreateLauncher();
var
  LauncherPath: string;
  LauncherContent: string;
  PythonwPath: string;
  AppPath: string;
begin
  PythonwPath := ExpandConstant('{localappdata}') + '\Programs\Python\Python311\pythonw.exe';
  AppPath := ExpandConstant('{app}') + '\buki.py';
  LauncherPath := ExpandConstant('{app}') + '\launch.vbs';
  LauncherContent := 'Set sh = CreateObject("WScript.Shell")' + #13#10 +
                     'sh.Run """' + PythonwPath + '"" """ + AppPath + """", 0, False' + #13#10;
  SaveStringToFile(LauncherPath, LauncherContent, False);
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
    CreateLauncher();
end;
