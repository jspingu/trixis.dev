const tickRate = 120
const tickTime = 1 / tickRate

const configs = [
    { dividend: 51, divisor: 12, frequency: 77 },
    { dividend: 81, divisor: 18, frequency: 72 },
    { dividend: 68, divisor: 27, frequency: 83 },
    { dividend: 60, divisor: 17, frequency: 100 },
    { dividend: 98, divisor: 100, frequency: 100 },
    { dividend: 87, divisor: 20, frequency: 75 },
]

let primaryColor
let trailContainer
let dividend
let divisor
let frequency
let trails = []
let elapsedTime = 0

function createTrail(x, y, trailHeadRadius, trailPointCount) {
    let trailHead = document.createElementNS("http://www.w3.org/2000/svg", "circle")
    trailHead.setAttribute("r", trailHeadRadius)
    trailHead.setAttribute("fill", primaryColor)

    let trail = {
        x: x, y: y,
        head: trailHead,
        headRadius: trailHeadRadius,
        points: [...Array(trailPointCount).keys()].map(_ => ({ x: 0, y: 0 })),
        shapes: [...Array(trailPointCount - 1).keys()].map(i => {
            let shape = document.createElementNS("http://www.w3.org/2000/svg", "polygon")
            shape.setAttribute("fill", primaryColor)
            shape.setAttribute("opacity", `${(i + 1) / (trailPointCount - 1)}`)
            return shape
        }),
    }

    trailContainer.append(trail.head)
    trail.shapes.forEach(shape => trailContainer.append(shape))
    trails.push(trail)
}

function updateTrails() {
    elapsedTime += tickTime

    trails.forEach(trail => {
        let ratio = dividend.value / divisor.value
        let baseFreq = frequency.value
        let center = trailContainer.getBoundingClientRect().width / 2

        trail.x = center + Math.cos(elapsedTime * baseFreq) * (center / 2 - 1) + Math.cos(elapsedTime * baseFreq * ratio) * (center / 2 - 1)
        trail.y = center + Math.sin(elapsedTime * baseFreq) * (center / 2 - 1) + Math.sin(elapsedTime * baseFreq * ratio) * (center / 2 - 1)

        trail.head.setAttribute("cx", trail.x)
        trail.head.setAttribute("cy", trail.y)

        trail.points.shift()
        trail.points.push({ x: trail.x, y: trail.y })

        trail.shapes.reduce((begin, shape, i) => {
            let orthogonal = {
                x: trail.points[i].y - trail.points[i + 1].y,
                y: trail.points[i + 1].x - trail.points[i].x
            }

            let len = Math.sqrt(orthogonal.x * orthogonal.x + orthogonal.y * orthogonal.y)
            orthogonal.x = orthogonal.x / len
            orthogonal.y = orthogonal.y / len

            let width = trail.headRadius * (i + 1) / (trail.points.length - 1)

            let end = {
                x1: trail.points[i + 1].x + orthogonal.x * width,
                x2: trail.points[i + 1].x - orthogonal.x * width,
                y1: trail.points[i + 1].y + orthogonal.y * width,
                y2: trail.points[i + 1].y - orthogonal.y * width,
            }

            shape.setAttribute("points", `${begin.x1},${begin.y1} ${end.x1},${end.y1} ${end.x2},${end.y2} ${begin.x2},${begin.y2}`)
            return end
        }, {
            x1: trail.points[0].x, x2: trail.points[0].x,
            y1: trail.points[0].y, y2: trail.points[0].y
        })
    })
}

document.addEventListener("DOMContentLoaded", () => {
    let trailsElem = document.getElementById("trails")

    if (!trailsElem)
        return

    trailContainer = document.createElementNS("http://www.w3.org/2000/svg", "svg")
    trailContainer.id = "trail-container"
    trailsElem.append(trailContainer)

    dividend = document.getElementById("slider-dividend")
    divisor = document.getElementById("slider-divisor")
    frequency = document.getElementById("slider-frequency")

    let initConfig = configs[Math.floor(Math.random() * configs.length)]
    dividend.value = initConfig.dividend
    divisor.value = initConfig.divisor
    frequency.value = initConfig.frequency

    primaryColor = getComputedStyle(document.body).getPropertyValue("--primary-color");
    createTrail(0, 0, 2, 128)

    setInterval(updateTrails, 1000 / tickRate)
})
