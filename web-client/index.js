'use strict';

function myLog(str)
{
    var now = new Date();
    var ss = document.getElementById("myLog").innerHTML;
    ss = now.HHMMSS() + ' ' + str + "<br>" + ss;
    document.getElementById("myLog").innerHTML = ss;
}

Date.prototype.HHMMSS = function () {
    var dd = this.getDate();
    var hh = this.getHours();
    var mm = this.getMinutes();
    var ss = this.getSeconds();

    return (dd > 9 ? '' : '0') + dd + '-' + ([
        (hh > 9 ? '' : '0') + hh,
        (mm > 9 ? '' : '0') + mm,
        (ss > 9 ? '' : '0') + ss
         ].join(':'));
}

let ws = {}

function connect() {
    ws = new WebSocket("ws://127.0.0.1:8266/");

    ws.onopen = function () {
        myLog('ws connected')
    };

    ws.onmessage = function (e) {
        var d = e.data
        // myLog(d)
        if (d == '[PING]')
            ws.send('[PONG]')
        else {
            if (d[0] == '{') {
                d = JSON.parse(d)
                document.getElementById("textArea").value += d.txt

                localStorage.setItem('txt_stored', document.getElementById("textArea").value);
            }
        }

    }

    ws.onclose = function (e) {
        // myLog('ws closed. retrying...', e.reason);
        setTimeout(function () {
            connect();
        }, 2000);
    };

    ws.onerror = function (e) {
        myLog('ws error', e.message);
        ws.close();
    };
}

let txt_stored = localStorage.getItem('txt_stored');
if (txt_stored)
    document.getElementById("textArea").value = txt_stored

myLog("opening... 请稍等...")
connect();

let lastGeneratePosition = -1

function sendText() {
    let msg = {}
    msg.op = 'GET'
    let txt = document.getElementById("textArea").value
    lastGeneratePosition = txt.length
    msg.txt = txt.substr(Math.max(0, txt.length - 511)) // 512-1=511
    ws.send(JSON.stringify(msg))
}

function rewriteText() {
    let txt = document.getElementById("textArea").value
    if (lastGeneratePosition == -1) {
        if (txt.length == 0)
            lastGeneratePosition = 0
    }
    if (lastGeneratePosition != -1) {
        txt = txt.substr(0, lastGeneratePosition)
        document.getElementById("textArea").value = txt
    }
    sendText()
}

document.onkeydown = function (event) {
    if (event.altKey) {
        if (event.keyCode === 81) {
            sendText()
            return false
        }
        if (event.keyCode === 69) {
            rewriteText()
            return false
        }
    }
}