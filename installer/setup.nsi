Unicode true
Target amd64-unicode
SetCompressor /SOLID lzma
!include "MUI2.nsh"
!include "nsDialogs.nsh"
!include "LogicLib.nsh"

!define APPNAME "모자이크 생성기"
!define REGKEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\MosaicGenerator"

Name "${APPNAME}"
OutFile "MosaicSetup.exe"
InstallDir "C:\모자이크생성기"
InstallDirRegKey HKLM "${REGKEY}" "InstallLocation"
RequestExecutionLevel admin
ShowInstDetails hide
ShowUninstDetails hide
BrandingText "${APPNAME}"

!define MUI_ICON "app.ico"
!define MUI_UNICON "app.ico"
!define MUI_ABORTWARNING
!define MUI_ABORTWARNING_TEXT "설치를 그만둘까요?"

!define MUI_WELCOMEPAGE_TITLE "${APPNAME} 설치"
!define MUI_WELCOMEPAGE_TEXT "영상의 원본 모자이크 위에 자연스러운 블러를 덧대고, 일본어·영어 대사를 한국어 자막으로 만들어 주는 프로그램이에요.$\r$\n$\r$\n설치하면서 필요한 프로그램 묶음과 AI 부품(수 GB)을 인터넷에서 받아요. 인터넷 속도에 따라 10~30분쯤 걸려요.$\r$\n$\r$\n모든 것은 고른 설치 폴더 안에만 들어가요.$\r$\n$\r$\n[다음]을 눌러 주세요."
!insertmacro MUI_PAGE_WELCOME

!define MUI_DIRECTORYPAGE_TEXT_TOP "설치할 폴더를 골라 주세요. 작업 파일과 AI 모델이 이 폴더에 쌓이니 여유 공간이 넉넉한 드라이브(50GB 이상 권장)를 골라 주세요."
!insertmacro MUI_PAGE_DIRECTORY

Page custom ModePage ModeLeave

!define MUI_INSTFILESPAGE_FINISHHEADER_TEXT "설치가 끝났어요"
!define MUI_INSTFILESPAGE_FINISHHEADER_SUBTEXT "바탕화면의 '${APPNAME}' 아이콘으로 켜면 돼요."
!insertmacro MUI_PAGE_INSTFILES

!define MUI_FINISHPAGE_TITLE "설치 완료"
!define MUI_FINISHPAGE_TEXT "바탕화면과 시작 메뉴에 '${APPNAME}' 아이콘을 만들었어요.$\r$\n$\r$\n켜면 검은 창 없이 브라우저가 열리고, 화면 오른쪽 아래 시계 옆에 작은 아이콘이 생겨요. 끌 때는 그 아이콘을 오른쪽 클릭 → [끄기].$\r$\n$\r$\n처음 켤 때는 AI 모델을 받느라 몇 분 더 걸릴 수 있어요."
!define MUI_FINISHPAGE_RUN
!define MUI_FINISHPAGE_RUN_TEXT "지금 켜기"
!define MUI_FINISHPAGE_RUN_FUNCTION RunApp
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "Korean"

Var Mode
Var RadB
Var RadL
Var RadS

Function ModePage
  !insertmacro MUI_HEADER_TEXT "무엇을 쓰실 건가요?" "이 PC에서 쓸 기능을 골라 주세요. 고른 기능에 필요한 AI 모델만 받아요."
  nsDialogs::Create 1018
  Pop $0
  ${NSD_CreateRadioButton} 10u 10u 280u 14u "블러 + 자막 둘 다"
  Pop $RadB
  ${NSD_CreateRadioButton} 10u 30u 280u 14u "블러만 (모자이크 덧대기)"
  Pop $RadL
  ${NSD_CreateRadioButton} 10u 50u 280u 14u "자막만 (받아쓰기·번역)"
  Pop $RadS
  ${NSD_CreateLabel} 10u 76u 280u 40u "나중에 바꾸고 싶으면 다시 설치하면 돼요. 이미 받은 것은 다시 받지 않아요."
  Pop $0
  ${If} $Mode == "blur"
    ${NSD_Check} $RadL
  ${ElseIf} $Mode == "subs"
    ${NSD_Check} $RadS
  ${Else}
    ${NSD_Check} $RadB
  ${EndIf}
  nsDialogs::Show
FunctionEnd

Function ModeLeave
  StrCpy $Mode "both"
  ${NSD_GetState} $RadL $0
  ${If} $0 == ${BST_CHECKED}
    StrCpy $Mode "blur"
  ${EndIf}
  ${NSD_GetState} $RadS $0
  ${If} $0 == ${BST_CHECKED}
    StrCpy $Mode "subs"
  ${EndIf}
FunctionEnd

!macro Step NAME LABEL
  DetailPrint "${LABEL}"
  SetDetailsPrint listonly
  nsExec::ExecToLog '"$SYSDIR\cmd.exe" /c ""$INSTDIR\setup_steps.bat" ${NAME} $Mode"'
  Pop $0
  SetDetailsPrint both
  ${If} $0 != "0"
    MessageBox MB_ICONSTOP "설치 중에 문제가 생겼어요 ($\"${LABEL}$\" 단계).$\r$\n$\r$\n[자세히 보기]를 눌러 나온 글을 캡처해서 보내 주세요.$\r$\n인터넷 연결을 확인한 뒤 설치 파일을 다시 실행하면 이어서 설치돼요."
    SetDetailsView show
    Abort
  ${EndIf}
!macroend

Section "Install"
  SetShellVarContext current
  SetOutPath "$INSTDIR"
  File "setup_steps.bat"
  File "app.ico"
  !insertmacro Step "vc" "1/5  윈도우 기본 부품 확인 중…"
  !insertmacro Step "python" "2/5  파이썬 준비 중…"
  !insertmacro Step "fetch" "3/5  프로그램 받는 중…"
  !insertmacro Step "packages" "4/5  필요한 프로그램 묶음 설치 중… (가장 오래 걸려요, 10~20분)"
  !insertmacro Step "finish" "5/5  마무리 중…"
  CreateShortCut "$DESKTOP\${APPNAME}.lnk" "$INSTDIR\python\pythonw.exe" '"$INSTDIR\launcher.pyw"' "$INSTDIR\app.ico" 0
  CreateShortCut "$SMPROGRAMS\${APPNAME}.lnk" "$INSTDIR\python\pythonw.exe" '"$INSTDIR\launcher.pyw"' "$INSTDIR\app.ico" 0
  WriteUninstaller "$INSTDIR\제거하기.exe"
  WriteRegStr HKLM "${REGKEY}" "DisplayName" "${APPNAME}"
  WriteRegStr HKLM "${REGKEY}" "DisplayIcon" "$INSTDIR\app.ico"
  WriteRegStr HKLM "${REGKEY}" "InstallLocation" "$INSTDIR"
  WriteRegStr HKLM "${REGKEY}" "UninstallString" '"$INSTDIR\제거하기.exe"'
  WriteRegStr HKLM "${REGKEY}" "Publisher" "bnn"
  WriteRegDWORD HKLM "${REGKEY}" "NoModify" 1
  WriteRegDWORD HKLM "${REGKEY}" "NoRepair" 1
  DetailPrint "설치 끝!"
SectionEnd

Function RunApp
  ; 관리자 권한이 아닌 보통 권한으로 켜기 (탐색기를 거쳐 바로가기 실행)
  Exec '"$WINDIR\explorer.exe" "$SMPROGRAMS\${APPNAME}.lnk"'
FunctionEnd

Function un.onInit
  MessageBox MB_YESNO|MB_ICONQUESTION "${APPNAME}를 지울까요?$\r$\n$\r$\n설치 폴더($INSTDIR) 안의 작업 기록과 그 안에 저장된 결과 영상도 함께 지워져요.$\r$\n(따로 정한 결과 저장 폴더의 영상은 그대로 남아요)" IDYES +2
  Abort
FunctionEnd

Section "Uninstall"
  SetShellVarContext current
  ; 켜져 있으면 먼저 끔 (이 폴더 안 프로그램만)
  nsExec::Exec 'powershell -NoProfile -Command "Get-Process | Where-Object { $$_.Path -like \"$INSTDIR\*\" } | Stop-Process -Force"'
  Delete "$DESKTOP\${APPNAME}.lnk"
  Delete "$SMPROGRAMS\${APPNAME}.lnk"
  RMDir /r "$INSTDIR"
  DeleteRegKey HKLM "${REGKEY}"
SectionEnd
