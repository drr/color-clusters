import * as d3 from "https://cdn.skypack.dev/d3@7.4";
import * as d3ColorDiff from "https://cdn.skypack.dev/d3-color-difference"
export * as d3 from "https://cdn.skypack.dev/d3@7.4"; // for REPL debugging

const colorDiff = d3ColorDiff.differenceCiede2000;      // Use the same difference function everywhere

// Define the percentiles for which we should count the number of unique colors required to cover this percent of pixels
// TODO: experiment with this as a collection class instead of element (ability to iterate, work at the level of the name
//       of the percentile, etc.)
class Percentile {
    static defaultPtiles = [10, 50, 75, 90, 99, 99.9, 100];
    name;               // Nth percentile
    cumulativeCount;    // Running total of elements
    threshold;          // cumulativeCount must exceed threshold to reach the Nth percentile

    // Return an array of Percentile objects with thresholds relative to populationSize
    static Factory(populationSize, ptilesTemplate = Percentile.defaultPtiles) {
        const percentiles = new Array();
        for (const p of ptilesTemplate) {
            percentiles.push(new Percentile(p, populationSize));
        }
        return percentiles;
    }

    constructor(ptile, populationSize) {
        this.name = ptile;
        this.cumulativeCount = 0;
        this.threshold = Math.floor(populationSize * (ptile / 100));
    }
}

// Aggregate of position and color
class Pixel {
    position = null;    // [x, y] array
    rgbColor = null;
    lchColor = null;
    assignedCluster = null;

    constructor(position, rgb = null, lch = null) {
        this.position = position;
        this.rgbColor = rgb;
        this.lchColor = lch;
        //console.info(`Created pixel at ${position}: ${rgb}, ${lch}`);
    }

    toString() {
        return `[${this.position.toString()}]: ${this.rgbColor.formatHex()}`;
    }

    // Use the cluster add/remove() methods so the cluster can track if recalculation is needed
    setCluster(cluster) {
        console.assert(cluster, `Expected non-null cluster: ${cluster}`);
        if (this.assignedCluster) {
            this.assignedCluster.removeMember(this);
        }
        cluster.addMember(this);
        this.assignedCluster = cluster;
    }

    getCluster() {
        return this.assignedCluster;
    }

    // Use this to experiment with measuring distance based on different color spaces
    get color() {
        return this.rgbColor.formatHex();
    }

}

/*
 * Cluster tracks the centroid for a single cluster and the members assigned to this cluster
 * It tracks damage when updates are made to the centroid or membership.
 *   centroid = '#rrggbb' hex string of a color
 * 
 * Members can be any type of object that responds properly to arrayName.includes(member)
 */

class Cluster {
    centroid = 'should never be empty'; // A color, not a Pixel
    members = [];                       // Array of Pixel members
    changedCentroid = false;
    changedMembers = false;
    history = [];                       // Track how the cluster changes over time

    constructor(centroid) {
        this.centroid = centroid;
    }

    toString() {
        return this.centroid.toString();
    }

    clearChanged() {
        this.changedCentroid = false;
        this.changedMembers = false;
    }

    addMember(member) {
        if (this.members.includes(member)) {
            console.warn(`Cluster already has ${member}`);
            return false;
        }
        this.members.push(member);
        this.changedMembers = true;
        //console.info(`Added member ${member} to cluster ${this}`);
        return true;
    }

    removeMember(member) {
        const index = this.members.indexOf(member);
        if (index < 0) {
            console.warn(`Cluster did not include ${member}`);
            return false;
        }
        const newMemberList = this.members.slice(0, index).concat(this.members.slice(index + 1));
        //console.info(`Removing ${member} at index ${index} from ${this}`);
        console.assert(newMemberList.length = this.members.length - 1);
        //console.info(`Length was ${this.members.length} now ${newMemberList.length}`);
        this.members = newMemberList;
        this.changedMembers = true;
        return true;

    }

    getCentroid() {
        return this.centroid;
    }

    /*
     * Calculate the mean of the members in a cluster. Adopt the mean as the new centroid
     * if it has changed by more than tolerance. Does not change membership.
     */
    updateCentroid(tolerance = 1) {
        console.groupCollapsed(`Start updating cluster ${this}`);

        let updated = false;
        const currentCentroid = this.centroid;
        const meanRGB = this.calculateMean();
        const distance = colorDiff(meanRGB, currentCentroid);
        if (distance > tolerance) {
            //console.info(`New ${meanRGB} is better than ${currentCentroid} by ${distance.toFixed(2)}`);
            updated = true;
            this.centroid = meanRGB;
            this.changedCentroid = true;
        }
        console.groupEnd();
        console.info(`Cluster ${this} - Updated: ${updated} Members: ${this.changedMembers}(${this.members.length}), was ${currentCentroid}, ${meanRGB} ${distance.toFixed(2)}`)
        this.addHistory(updated, currentCentroid, meanRGB, distance);
        return updated;
    }

    get history() {
        return this.history;
    }

    // Track changes to the cluster that could be of interest in understanding K-Means progress
    addHistory(wasUpdated, old, mean, delta) {
        // Run through and calculate min and max color difference as part of creating a history entry
        let minDistance = Infinity;
        let maxDistance = 0;
        for (const pixel of this.members) {
            const distance = colorDiff(pixel.color, this.centroid);
            if (distance < minDistance) minDistance = distance;
            if (distance > maxDistance) maxDistance = distance;
        }

        const historyEntry = {
            centroid: this.centroid,
            changed: wasUpdated,
            count: this.members.length,
            oldCentroid: old,
            candidateCentroid: mean,
            delta: delta,
            minDistance: minDistance,
            maxDistance: maxDistance,
        };
        this.history.push(historyEntry);
    }

    /*
     * Calculate the mean RGB of the members and return as '#rrggbb' formatted hex string
     *
     */
    calculateMean() {
        // Need to make an array of colors, not Pixels
        const colors = this.members.map(pixel => pixel.color);
        const count = colors.length;
        // Special case if there are no members. Return a random color to re-seed the cluster
        if (count === 0) {
            const rand255 = () => Math.floor(Math.random() *255);
            const newColor = d3.rgb(rand255(), rand255(), rand255()).formatHex();
            console.log(`Member count is zero, creating a random color ${newColor}`);
            return newColor;
        }

        let R = 0;
        let G = 0;
        let B = 0;
        for (const color of colors) {
            const r = Number.parseInt(color.substring(1, 3), 16);
            const g = Number.parseInt(color.substring(3, 5), 16);
            const b = Number.parseInt(color.substring(5, 7), 16);
            console.assert(d3.rgb(r, g, b).formatHex() === color);
            R += (r * r);
            G += (g * g);
            B += (b * b);
        }
        R = Math.round(Math.sqrt(R / count));
        console.assert(Number.isInteger(R));
        G = Math.round(Math.sqrt(G / count));
        console.assert(Number.isInteger(G));
        B = Math.round(Math.sqrt(B / count));
        console.assert(Number.isInteger(B));
        const newColor = d3.rgb(R, G, B).formatHex();
        //console.info(`Ave: ${R} ${G} ${B} hex ${newColor}`);
        return newColor;
    }

}

export default class AnalyzableImage {
    url = null;         // Image src
    name = null;        // Name to use as id for this image
    image = null;       // raw Image()

    canvas = null;      // Offscreen canvas from which we get pixel data
    context = null;     // Context of canvas
    imageData = null;   // ImageData() object for this image
    pixelCount = null;  // Count of pixels (RGBA elements)
    pixelArray = null;  // Array of Pixel objects
    colorFreqMap = null;// Count of times each unique RGB color appears
    uniqueColorCount = null;

    percentiles = null;
    // Define the exact-match topN stat for which we should calculate cumulative percent of pixels covered by N
    topN = new Map([
        [1, -1],
        [5, -1],
        [10, -1],
        [100, -1],
        [1000, -1],
        [10_000, -1],
        [100_000, -1],
    ]);

    clusters = null;    // Array of K-Means Cluster objects

    // Learning JS: resist the temptation to make the constructor async because anything
    // other than the implicit return of 'this' breaks a lot of expectations. The two
    // main alternatives are either a static Class.factoryCreate() method that internally
    // invokes the constructur, or a post-constructor instance.initialize() method that does the
    // async work
    constructor(url, name = null) {
        this.url = url;
        this.name = name;
    }

    // Return a promise for loading the image and setting it to the specified maxWidth. After the promise
    // completes, operations can be done on the image
    async loadImage(maxWidth = 0) {
        this.image = new Image();
        this.image.src = this.url;
        if (this.name) this.image.id = this.name;

        const loadingPromise = this.image.decode()
            .then(response => {
                console.log(`Response after decode() '${response}'`);
                console.log(`Image: ${this.image} complete: ${this.image.complete}`);
                if (maxWidth > 0) {
                    this.image.style = `max-width: ${Math.min(this.image.width, maxWidth)}px`
                }
            })
            /* .then(response => {
                console.log(`Response in second phase: '${response}'`)
                this.buildColorData();
            }) */
            .catch(reason => {
                console.error(`I do not understand this call path ${reason}`);

            });

        return loadingPromise;
    }

    // Build the datasets that will be used to analyze the image
    buildColorData() {

        // Local copies for convenience
        const width = Math.floor(this.image.width);
        const height = Math.floor(this.image.height);

        // Create an offscreen canvas we'll use to capture the image data
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        canvas.id = this.name;  // For debugging

        // Render into the canvas's drawing context, then access the raw data
        const context = canvas.getContext('2d');
        context.drawImage(this.image, 0, 0, width, height);
        const imageData = context.getImageData(0, 0, width, height);

        // Data is formatted as RGBA individual byte elements in the array. Capture it as pixels
        console.assert(imageData.colorSpace === 'srgb');
        const byteStride = 4;
        const imgByteData = imageData.data;
        const pixelCount = imgByteData.length / byteStride;
        const pixelArray = new Array();
        const colorFreqMap = new Map();

        // Convert the raw array of bytes into an array of pixels.
        // While we traverse this, also build a frequency map that counts occurences of unique colors
        console.groupCollapsed('Build pixel array');
        for (let i = 0, xPos = 0, yPos = 0; i < imgByteData.length; i += byteStride) {
            const r = imgByteData[i];
            const g = imgByteData[i + 1];
            const b = imgByteData[i + 2];
            const a = imgByteData[i + 3];

            const rgbColor = d3.rgb(r, g, b);
            const lchColor = d3.lch(rgbColor);
            const pixel = new Pixel([xPos, yPos], rgbColor, lchColor);
            pixelArray.push(pixel);

            // Learning JS: Use a simple primitive type (here a string) as the Map key, not an object
            const hexColor = rgbColor.formatHex();
            if (colorFreqMap.has(hexColor)) {
                colorFreqMap.set(hexColor, colorFreqMap.get(hexColor) + 1);
            } else {
                colorFreqMap.set(hexColor, 1);
            }

            xPos++;
            if (xPos === width) xPos = 0;
            if (xPos === 0) yPos++;
        }
        console.groupEnd();

        // Save our work into the object instance
        this.canvas = canvas;
        this.context = context;
        this.imageData = imageData;
        this.pixelArray = pixelArray;
        this.pixelCount = pixelCount;
        this.colorFreqMap = colorFreqMap;
        this.uniqueColorCount = colorFreqMap.size;

        return;
    }

    // Calculate topN and percentiles for the number of unique colors required to cover the total pixel population
    calculateSummaryStatistics() {
        // Learning JS:
        // The ...spread operator is the equiv of Array.from(colorFreqMap.entries())
        // which is made into a new 2D Array of all [key, value] array pairs because it is surrounded by []
        // then it sorts them based on comparing the values (index[1] in the 2D array)
        // and then it makes a new Map(fromArray)
        const sortedColorFreq = new Map([...this.colorFreqMap.entries()].sort((a, b) => b[1] - a[1]));
        this.colorFreqMap = sortedColorFreq;

        const percentiles = Percentile.Factory(this.pixelCount);

        // Scan through the sorted frequency list, populating statistics as we go
        let targetPtileIndex = 0;   // The entry in the Percentiles array we are searching for next
        let cumColors = 0;          // How many of the total colors have we seen so far
        let cumCoveredPixels = 0;   // How many of the total pixels have we seen so far

        // TODO: this doesn't work properly if a single color sample crosses multiple ptile thresholds
        for (const [hex, freq] of sortedColorFreq) {
            cumColors++;
            cumCoveredPixels += freq;
            const pcntOfPixels = cumCoveredPixels / this.pixelCount;
            if (this.topN.has(cumColors)) this.topN.set(cumColors, pcntOfPixels);
            if (cumCoveredPixels >= percentiles[targetPtileIndex].threshold) {
                percentiles[targetPtileIndex].cumulativeCount = cumColors;
                percentiles[targetPtileIndex].datapoint = {
                    hex: hex,
                    freq: freq,
                    cumCoveredPixels: cumCoveredPixels,
                };
                targetPtileIndex++;
            }
        }
        console.assert(cumCoveredPixels == this.pixelCount);
        console.groupCollapsed(`Summary stats`);
        for (const [n, pcnt] of this.topN) {
            if (pcnt < 0) break;
            console.log(`top ${n.toLocaleString()} colors is ${(pcnt * 100).toPrecision(4)}% of pixels`);
        }

        for (const [i, ptile] of percentiles.entries()) {
            console.log(`pixel ptile ${ptile.name}: cumulativeCount ${ptile.cumulativeCount.toLocaleString()} (${(ptile.cumulativeCount / this.uniqueColorCount * 100).toPrecision(3)}% of color) cumCoveredPixels ${ptile.datapoint.cumCoveredPixels.toLocaleString()} (${ptile.datapoint.cumCoveredPixels - ptile.threshold} over threshold)`);
        }
        console.groupEnd();

        this.percentiles = percentiles;
        return;
    }

    describe() {
        const i = this.image;
        console.log(this);
        console.log(`Name '${this.name}' size ${i.width}x${i.height} has ${this.colorFreqMap.size.toLocaleString()} unique hex colors`);
        console.log(this.colorFreqMap);
        console.log(i.style);
        console.log(i);
    }

    reviewHistory() {
        const maxes = new Array();
        for (const cluster of this.clusters) {
            console.log(cluster);
            cluster.history.forEach((elem, index) => {
                console.log(`${index}: ${elem.centroid} count ${elem.count} maxD ${elem.maxDistance.toFixed(2)} minD ${elem.minDistance.toFixed(2)} ${elem.changed} ${elem.delta.toFixed(2)} ${elem.oldCentroid} -> ${elem.candidateCentroid}`);
            });
            maxes.push(Math.round(cluster.history[cluster.history.length -1].maxDistance));
        }
        console.log(`Final maxes:`);
        const table = new Array();
        table.push(Array.from(this.clusters.map(cluster => cluster.centroid)));
        table.push(Array.from(this.clusters.map(cluster => cluster.members.length)));
        table.push(maxes);
        console.table(table, this.clusters.length);
    }

    // Let's look at how far the centroids are apart from each other
    clusterDistances() {
        for (const srcCluster of this.clusters) {
            const srcCentroid = srcCluster.centroid;
            console.log(srcCluster);
            for (const dstCluster of this.clusters) {
                if (dstCluster === srcCluster) continue;
                const dstCentroid = dstCluster.centroid;
                const distance = colorDiff(srcCentroid, dstCentroid);
                console.log(`${distance.toFixed(2)} -> ${dstCentroid}`);
            }
        }
    }

    // Create a matrix of color differences between clusters
    calculateClusterDistranceMatrix() {
        const clusterCount = this.clusters.length;
        const matrix = new Array(clusterCount);
        for (let row = 0; row < clusterCount; row++) matrix[row] = new Array(clusterCount);

        for (let x = 0; x < clusterCount; x++) {
            const srcCentroid = this.clusters[x].centroid;
            matrix[x][x] = 0;
            for (let y = x+1; y < clusterCount; y++) {
                const dstCentroid = this.clusters[y].centroid;
                const distance = Math.round(colorDiff(srcCentroid, dstCentroid));
                //console.info(`[${x}, ${y}]: ${distance} ${srcCentroid} -> ${dstCentroid}`);
                matrix[x][y] = distance;
                matrix[y][x] = distance;
            }
        }
        console.table(matrix, clusterCount);
    }

    async calculateKMeans(clusterCount = 8, maxRounds = 10, tolerance = 3) {
        // Use the color data from a sampling of the PixelArray to seed the initial centroids in our clusters
        this.clusters = this.samplePixelArray(clusterCount).map(sample => new Cluster(sample.color));

        for (let rounds = 0; rounds < maxRounds; rounds++) {
            console.groupCollapsed(`K-Means Iteration round: ${rounds}`);
            let changed = false;
            let changedCentroidCount = 0;
            let changedPixelCount = this.assignPixelsToClusters(tolerance);
            for (const cluster of this.clusters) {
                cluster.updateCentroid(tolerance);
                if (cluster.changedCentroid || cluster.changedMembers) {
                    changed = true;
                    (cluster.changedCentroid) ? changedCentroidCount++ : false;
                    cluster.clearChanged();
                }
            }
            console.groupEnd();
            console.log(`iteration ${rounds}: assigned ${changedPixelCount} pixels, changed ${changedCentroidCount} centroids`);
            if (!changed) break;

        }
    }

    /*
     * Return an array with numSamples members randomly chosen from the pixelArray dataset
     */
    samplePixelArray(numSamples) {
        const chunkSize = Math.floor(this.pixelArray.length / numSamples);
        let samples = [];

        for (let i = 0; i < numSamples; i++) {
            samples.push(this.pixelArray[(i * chunkSize) + (Math.floor(Math.random() * chunkSize))]);
        }
        console.log(`Created samples: ${samples}`);
        return samples;
    }

    /*
     * Assign a pixel to a cluster if the distance to the centroid of that cluster is nearer
     * than the distance in the current cluster, by greater than the tolerance amount.
     */
    assignPixelsToClusters(tolerance = 1) {
        console.groupCollapsed('Start assigning');
        let changeCount = 0;

        for (const pixel of this.pixelArray) {
            let bestDistance = Infinity;
            let bestCluster = pixel.getCluster();
            // TODO: I think this test can be improved to also check for if the centroid changed?
            if (bestCluster) {
                bestDistance = colorDiff(pixel.rgbColor, bestCluster.getCentroid());
                bestDistance = (bestDistance > tolerance) ? bestDistance - tolerance : 0;
            }
            for (const cluster of this.clusters) {
                const distance = colorDiff(pixel.rgbColor, cluster.getCentroid());
                if (Number.isNaN(distance) || distance === Infinity) {
                    console.log(`Analyzed ${pixel} with ${distance} to ${cluster}`);
                }
                if (distance < bestDistance) {
                    bestCluster = cluster;
                    bestDistance = (distance > tolerance) ? distance - tolerance : 0;
                }
            }
            //console.log(`Analyzed ${pixel}: ${bestDistance} to cluster ${bestCluster}`);
            if (bestCluster !== pixel.getCluster()) {
                pixel.setCluster(bestCluster);
                changeCount++;
            } else {
                // console.log(`Already had best cluster for ${pixel}: ${bestDistance} ${bestCluster}`);
                // console.log(pixel);
            }
        }
        console.groupEnd();
        console.log(`Finished assigning with ${changeCount} changes`);
        return changeCount;
    }

}

async function sampleRun(url = "./saul.jpg", name = "new saul") {
    console.log(d3.rgb(20, 30, 40).formatHex());
    let ns = new AnalyzableImage(url, name);
    let np = ns.loadImage(200);
    await np;
    console.log(`Now: ${ns} size ${ns.image.width}x${ns.image.height}`);
    document.body.appendChild(ns.image);
    //ns.buildColorData();
    ns.describe();

    console.log("pause here");
}

