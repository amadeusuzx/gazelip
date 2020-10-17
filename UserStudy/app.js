// element select

// Socket 

const socket = new WebSocket('ws://25.38.203.66:10130');
var currentTarget = null;
var container = document.querySelector("#container");
var playerctns = document.querySelector(".player-ctn");
// Music player
function createTrackItem(index, name, duration) {
    var trackItem = document.createElement('div');
    trackItem.setAttribute("class", "playlist-track-ctn");
    trackItem.setAttribute("id", "ptc-" + index);
    trackItem.setAttribute("data-index", index);
    document.querySelector(".playlist-ctn").appendChild(trackItem);

    var playBtnItem = document.createElement('div');
    playBtnItem.setAttribute("class", "playlist-btn-play");
    playBtnItem.setAttribute("id", "pbp-" + index);
    document.querySelector("#ptc-" + index).appendChild(playBtnItem);

    var btnImg = document.createElement('i');
    btnImg.setAttribute("class", "fas fa-play");
    btnImg.setAttribute("height", "40");
    btnImg.setAttribute("width", "40");
    btnImg.setAttribute("id", "p-img-" + index);
    document.querySelector("#pbp-" + index).appendChild(btnImg);

    var trackInfoItem = document.createElement('div');
    trackInfoItem.setAttribute("class", "playlist-info-track");
    trackInfoItem.innerHTML = name
    document.querySelector("#ptc-" + index).appendChild(trackInfoItem);

    var trackDurationItem = document.createElement('div');
    trackDurationItem.setAttribute("class", "playlist-duration");
    trackDurationItem.innerHTML = duration
    document.querySelector("#ptc-" + index).appendChild(trackDurationItem);
}

var listAudio = [
    {
        name: "Artist 1 - audio 1",
        file: "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-10.mp3",
        duration: "08:47"
    },
    {
        name: "Artist 2 - audio 2",
        file: "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-5.mp3",
        duration: "05:53"
    },
    {
        name: "Artist 3 - audio 3",
        file: "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-7.mp3",
        duration: "7:00"
    }
]

for (var i = 0; i < listAudio.length; i++) {
    createTrackItem(i, listAudio[i].name, listAudio[i].duration);
}
var indexAudio = 0;

function loadNewTrack(index) {
    var player = document.querySelector('#source-audio')
    player.src = listAudio[index].file
    document.querySelector('.title').innerHTML = listAudio[index].name
    this.currentAudio = document.getElementById("myAudio");
    this.currentAudio.load()
    this.toggleAudio()
    this.updateStylePlaylist(this.indexAudio, index)
    this.indexAudio = index;
}

var playListItems = document.querySelectorAll(".playlist-track-ctn");

for (let i = 0; i < playListItems.length; i++) {
    playListItems[i].addEventListener("click", getClickedElement.bind(this));
}

function getClickedElement(event) {
    for (let i = 0; i < playListItems.length; i++) {
        if (playListItems[i] == event.target) {
            var clickedIndex = event.target.getAttribute("data-index")
            if (clickedIndex == this.indexAudio) { // alert('Same audio');
                this.toggleAudio()
            } else {
                loadNewTrack(clickedIndex);
            }
        }
    }
}

document.querySelector('#source-audio').src = listAudio[indexAudio].file
document.querySelector('.title').innerHTML = listAudio[indexAudio].name


var currentAudio = document.getElementById("myAudio");

currentAudio.load()

currentAudio.onloadedmetadata = function () {
    document.getElementsByClassName('duration')[0].innerHTML = this.getMinutes(this.currentAudio.duration)
}.bind(this);

var interval1;

function toggleAudio() {

    if (this.currentAudio.paused) {
        document.querySelector('#icon-play').style.display = 'none';
        document.querySelector('#icon-pause').style.display = 'block';
        document.querySelector('#ptc-' + this.indexAudio).classList.add("active-track");
        this.playToPause(this.indexAudio)
        this.currentAudio.play();
    } else {
        document.querySelector('#icon-play').style.display = 'block';
        document.querySelector('#icon-pause').style.display = 'none';
        this.pauseToPlay(this.indexAudio)
        this.currentAudio.pause();
    }
}

function pauseAudio() {
    this.currentAudio.pause();
    clearInterval(interval1);
}

var timer = document.getElementsByClassName('timer')[0]

var barProgress = document.getElementById("myBar");


var width = 0;

function onTimeUpdate() {
    var t = this.currentAudio.currentTime
    timer.innerHTML = this.getMinutes(t);
    this.setBarProgress();
    if (this.currentAudio.ended) {
        document.querySelector('#icon-play').style.display = 'block';
        document.querySelector('#icon-pause').style.display = 'none';
        this.pauseToPlay(this.indexAudio)
        if (this.indexAudio < listAudio.length - 1) {
            var index = parseInt(this.indexAudio) + 1
            this.loadNewTrack(index)
        }
    }
}


function setBarProgress() {
    var progress = (this.currentAudio.currentTime / this.currentAudio.duration) * 100;
    document.getElementById("myBar").style.width = progress + "%";
}


function getMinutes(t) {
    var min = parseInt(parseInt(t) / 60);
    var sec = parseInt(t % 60);
    if (sec < 10) {
        sec = "0" + sec
    }
    if (min < 10) {
        min = "0" + min
    }
    return min + ":" + sec
}

var progressbar = document.querySelector('#myProgress')
progressbar.addEventListener("click", seek.bind(this));


function seek(event) {
    var percent = event.offsetX / progressbar.offsetWidth;
    this.currentAudio.currentTime = percent * this.currentAudio.duration;
    barProgress.style.width = percent * 100 + "%";
}

function forward() {
    this.currentAudio.currentTime = this.currentAudio.currentTime + 5
    this.setBarProgress();
}

function rewind() {
    this.currentAudio.currentTime = this.currentAudio.currentTime - 5
    this.setBarProgress();
}


function next() {
    if (this.indexAudio < listAudio.length - 1) {
        var oldIndex = this.indexAudio
        this.indexAudio++;
        updateStylePlaylist(oldIndex, this.indexAudio)
        this.loadNewTrack(this.indexAudio);
    }
}

function previous() {
    if (this.indexAudio > 0) {
        var oldIndex = this.indexAudio
        this.indexAudio--;
        updateStylePlaylist(oldIndex, this.indexAudio)
        this.loadNewTrack(this.indexAudio);
    }
}

function updateStylePlaylist(oldIndex, newIndex) {
    document.querySelector('#ptc-' + oldIndex).classList.remove("active-track");
    this.pauseToPlay(oldIndex);
    document.querySelector('#ptc-' + newIndex).classList.add("active-track");
    this.playToPause(newIndex)
}

function playToPause(index) {
    var ele = document.querySelector('#p-img-' + index)
    ele.classList.remove("fa-play");
    ele.classList.add("fa-pause");
}

function pauseToPlay(index) {
    var ele = document.querySelector('#p-img-' + index)
    ele.classList.remove("fa-pause");
    ele.classList.add("fa-play");
}


function toggleMute() {
    var btnMute = document.querySelector('#toggleMute');
    var volUp = document.querySelector('#icon-vol-up');
    var volMute = document.querySelector('#icon-vol-mute');
    if (this.currentAudio.muted == false) {
        this.currentAudio.muted = true
        volUp.style.display = "none"
        volMute.style.display = "block"
    } else {
        this.currentAudio.muted = false
        volMute.style.display = "none"
        volUp.style.display = "block"
    }
}


// Draggable Items


var activeItem = null;

var active = false;
var test = null;

container.addEventListener("mousedown", dragStart, false);
container.addEventListener("mouseup", dragEnd, false);
container.addEventListener("mousemove", drag, false);
function dragStart(e) {
    if (!e.target && !playerctns.contains(currentTarget)) {
        active = true;

        // this is the item we are interacting with
        activeItem = currentTarget;

        if (activeItem !== null) {
            if (!activeItem.xOffset) {
                activeItem.xOffset = 0;
            }

            if (!activeItem.yOffset) {
                activeItem.yOffset = 0;
            }

            if (e.type === "touchstart") {
                activeItem.initialX = e.touches[0].clientX - activeItem.xOffset;
                activeItem.initialY = e.touches[0].clientY - activeItem.yOffset;
            } else {
                activeItem.initialX = e.clientX - activeItem.xOffset;
                activeItem.initialY = e.clientY - activeItem.yOffset;
            }
        }
    } else if (e.target !== e.currentTarget && !playerctns.contains(e.target)) {

        active = true;

        // this is the item we are interacting with
        activeItem = e.target;

        if (activeItem !== null) {
            if (!activeItem.xOffset) {
                activeItem.xOffset = 0;
            }

            if (!activeItem.yOffset) {
                activeItem.yOffset = 0;
            }

            if (e.type === "touchstart") {
                activeItem.initialX = e.touches[0].clientX - activeItem.xOffset;
                activeItem.initialY = e.touches[0].clientY - activeItem.yOffset;
            } else {
                activeItem.initialX = e.clientX - activeItem.xOffset;
                activeItem.initialY = e.clientY - activeItem.yOffset;
            }
        }
    }
}

function dragEnd(e) {
    if (activeItem !== null) {
        activeItem.initialX = activeItem.currentX;
        activeItem.initialY = activeItem.currentY;
    }

    active = false;
    activeItem = null;
}

function drag(e) {
    if (active) {
        if (e.type === "touchmove") {
            e.preventDefault();

            activeItem.currentX = e.touches[0].clientX - activeItem.initialX;
            activeItem.currentY = e.touches[0].clientY - activeItem.initialY;
        } else {
            activeItem.currentX = e.clientX - activeItem.initialX;
            activeItem.currentY = e.clientY - activeItem.initialY;
        }

        activeItem.xOffset = activeItem.currentX;
        activeItem.yOffset = activeItem.currentY;

        setTranslate(activeItem.currentX, activeItem.currentY, activeItem);
    }
}

function setTranslate(xPos, yPos, el) {
    el.style.transform = "translate3d(" + xPos + "px, " + yPos + "px, 0)";
}


let clientX = -100;
let clientY = -100;
const innerCursor = document.querySelector(".cursor--small");

const initCursor = () => {
    // add listener to track the current mouse position
    document.addEventListener("mousemove", e => {
        clientX = e.clientX;
        clientY = e.clientY;
        currentTarget = document.elementFromPoint(clientX, clientY);
    });

    // transform the innerCursor to the current mouse position
    // use requestAnimationFrame() for smooth performance
    const render = () => {
        innerCursor.style.transform = `translate(${clientX}px, ${clientY}px)`;
        // if you are already using TweenMax in your project, you might as well
        // use TweenMax.set() instead
        // TweenMax.set(innerCursor, {
        //   x: clientX,
        //   y: clientY
        // });

        requestAnimationFrame(render);
    };
    requestAnimationFrame(render);
};

initCursor();
let lastX = 0;
let lastY = 0;
let isStuck = false;
let showCursor = false;
let group, stuckX, stuckY, fillOuterCursor;

const initCanvas = () => {
    const canvas = document.querySelector(".cursor--canvas");
    const shapeBounds = {
        width: 75,
        height: 75
    };
    paper.setup(canvas);

    const strokeColor = "rgba(113, 173, 230, 0.5)";
    const strokeWidth = 3;
    const segments = 8;
    const radius = 35;

    // we'll need these later for the noisy circle
    const noiseScale = 150; // speed
    const noiseRange = 30; // range of distortion
    let isNoisy = false; // state

    // the base shape for the noisy circle
    const polygon = new paper.Path.RegularPolygon(
        new paper.Point(0, 0),
        segments,
        radius
    );
    polygon.strokeColor = strokeColor;
    polygon.strokeWidth = strokeWidth;
    polygon.smooth();
    group = new paper.Group([polygon]);
    group.applyMatrix = false;

    const noiseObjects = polygon.segments.map(() => new SimplexNoise());
    let bigCoordinates = [];

    // function for linear interpolation of values
    const lerp = (a, b, n) => {
        return (1 - n) * a + n * b;
    };

    // function to map a value from one range to another range
    const map = (value, in_min, in_max, out_min, out_max) => {
        return (
            ((value - in_min) * (out_max - out_min)) / (in_max - in_min) + out_min
        );
    };

    // the draw loop of Paper.js 
    // (60fps with requestAnimationFrame under the hood)
    paper.view.onFrame = event => {
        // using linear interpolation, the circle will move 0.2 (20%)
        // of the distance between its current position and the mouse
        // coordinates per Frame
        lastX = lerp(lastX, clientX, 0.2);
        lastY = lerp(lastY, clientY, 0.2);
        group.position = new paper.Point(lastX, lastY);
    }
}

initCanvas();

function newBubble() {
    newItem = document.createElement("div");
    newItem.classList.add("item")
    w = Math.random() * 150
    newItem.style.width = w + "px";
    newItem.style.height = newItem.style.width;
    newItem.style.backgroundColor = `rgb(${Math.random() * 255}, ${Math.random() * 255}, ${Math.random() * 255})`;
    newItem.style.border = "10px solid rgba(136, 136, 136, .5)";
    newItem.style.top = (clientY - w / 2 - 10) + "px";
    newItem.style.left = (clientX - w / 2 - 10) + "px";
    container.appendChild(newItem);
}

socket.addEventListener('message', function (event) {
    indexArray = event.data.split(" ");
    if (playerctns.contains(currentTarget)) {
        for (i = 0; i < 3; i++) {
            result = Number(indexArray.pop());
            if ([4, 5, 6, 8].includes(result)) {
                switch (result) {
                    case 6:
                        console.log("play_music");
                        if (window.currentAudio.paused) {
                            toggleAudio();
                            break;
                        }
                    case 8:
                        console.log("stop_playing");
                        if (!window.currentAudio.paused) {
                            toggleAudio();
                            break;
                        }
                    case 5:
                        console.log("next_song");
                        next();
                        break;
                    case 4:
                        console.log("last_music")
                        previous();
                        break;
                    default:
                        break;
                }
                break;
            }
        }
    } else if (currentTarget.classList[0] === "item") {
        for (i = 0; i < 3; i++) {
            result = Number(indexArray.pop());
            if ([0, 2, 3, 7, 9, 10].includes(result)) {
                switch (result) {
                    case 0:
                        console.log("bigger");
                        currentTarget.style.width = (currentTarget.getBoundingClientRect().width * 1.5) + "px";
                        currentTarget.style.height = currentTarget.style.width;
                        break;
                    case 7:
                        console.log("smaller");
                        currentTarget.style.width = (currentTarget.getBoundingClientRect().width / 1.5) + "px";
                        currentTarget.style.height = currentTarget.style.width;
                        break;
                    case 2:
                        console.log("drag");
                        dragStart(new MouseEvent("mousedown", {
                            clientX: clientX, clientY: clientY
                        }))
                        break;
                    case 3:
                        console.log("drop");
                        dragEnd();
                        break;
                    case 9:
                        console.log("turn_blue");
                        currentTarget.style.backgroundColor = "rgb(0,0,255)";
                        break;
                    case 10:
                        console.log("turn_red")
                        currentTarget.style.backgroundColor = "rgb(255,0,0)"
                        break;
                    default:
                        break;
                }
                break;
            }
        }

    } else {

        for (i = 0; i < 3; i++) {
            result = indexArray.pop()
            if (result === "1") {

                newBubble();
                break;
            }
        }
    }
});