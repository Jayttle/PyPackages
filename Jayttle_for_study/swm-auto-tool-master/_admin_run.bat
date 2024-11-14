@ECHO OFF
setlocal EnableDelayedExpansion

PUSHD %~DP0 & cd /d "%~dp0"
%1 %2
mshta vbscript:createobject("shell.application").shellexecute("%~s0","goto :target","","runas",1)(window.close)&goto :eof
:target
:: -----------------------------------------
::
:: YOUR CODE HERE
::
python main.py

pause > nul
exit
