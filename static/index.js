const allPixels = document.getElementsByClassName("canvas-pixel")
const allRows = document.getElementsByClassName("canvas-row")
const canvasElement = document.getElementById("number-canvas")

let isClicked = false
let drawWeight = 255
let canvMatrix = new Array(28)
for (let i = 0; i < 28; i++) {
	let matRow = Array(28).fill(0)
	canvMatrix[i] = matRow
}

let intervalId = null

function getPixelElement(row, column) {
	if (row < 0 || row > 27) {
		return null
	}
	if (column < 0 || column > 27) {
		return null
	}
	let ind = row * 28 + column
	return allPixels[ind]
}

function addPixelEvents() {
	let elementCounter = 0
	for (let p of allPixels) {
		p.dataset.row = Math.floor(elementCounter / 28)
		p.dataset.column = elementCounter % 28
		p.dataset.drawValue = 0
		p.addEventListener("pointerdown", () => {
			return false;
		})
		p.addEventListener("pointerover", () => {
			if (isClicked) {
				let currRow = parseInt(p.dataset.row)
				let currColumn = parseInt(p.dataset.column)
				let neighbors = [getPixelElement(currRow-1, currColumn), getPixelElement(currRow+1, currColumn), getPixelElement(currRow, currColumn-1), getPixelElement(currRow, currColumn+1)]
				for (let n of neighbors) {
					if (n == null) {
						continue
					}
					n.classList.add("canvas-pixel-fill")
					let newWeight = Math.min(drawWeight + 30, 255)
					let nDrawValCalc = 255 - newWeight
					if (nDrawValCalc > n.dataset.drawValue) {
						n.dataset.drawValue = nDrawValCalc
						n.style.backgroundColor = `rgb(${newWeight},${newWeight},${newWeight})`
					}
				}
				p.classList.add("canvas-pixel-fill")
				let drawValCalc = 255 - drawWeight
				if (drawValCalc > p.dataset.drawValue) {
					p.dataset.drawValue = drawValCalc
					p.style.backgroundColor = `rgb(${drawWeight},${drawWeight},${drawWeight})`
				}
			}
		})
		elementCounter++
	}
}

function clearCanvas() {
	let filledPixels = document.querySelectorAll(".canvas-pixel-fill") // cannot use getElementsByClassName because it's a live list
	for (let p of filledPixels) {
		p.classList.remove("canvas-pixel-fill")
		p.style.backgroundColor = ""
		p.dataset.drawValue = "0"
	}
}

function submitNumber() {
	let filledPixels = document.getElementsByClassName("canvas-pixel-fill")
	for (let p of filledPixels) {
		let row = parseInt(p.dataset.row)
		let column = parseInt(p.dataset.column)
		let drawValue = parseInt(p.dataset.drawValue)
		canvMatrix[row][column] = drawValue
	}
	console.log(canvMatrix)
	fetch("/nn/number", {
		method: "POST",
		headers: {
			"Content-Type": "application/json;charset=UTF-8"
		},
		body: JSON.stringify({
			number: canvMatrix
		})
	})
}

function increaseDraw() {
	if (drawWeight < 0) {
		drawWeight = 0
		return
	}
	drawWeight -= 20
}


canvasElement.addEventListener("pointerdown", (event) => {
	event.preventDefault()
	isClicked = true
	intervalId = setInterval(increaseDraw, 5)
	console.log([isClicked, intervalId, drawWeight])
})

canvasElement.addEventListener("pointerup", () => {
	clearInterval(intervalId)
	isClicked = false
	drawWeight = 255
	console.log([isClicked, intervalId, drawWeight])
})


addPixelEvents()
