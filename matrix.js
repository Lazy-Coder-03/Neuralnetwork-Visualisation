// matrix.js - Robust Matrix Implementation

function normaliseMatrix(a) {
    if (!(a instanceof Matrix)) {
        throw new TypeError("normaliseMatrix expects a Matrix instance.");
    }

    let result = a.copy();
    let sum = 0;

    for (let i = 0; i < a.rows; i++) {
        for (let j = 0; j < a.cols; j++) {
            sum += Math.abs(result.data[i][j]);
        }
    }

    if (sum === 0) {
        console.warn("Warning: Normalisation skipped, matrix sum is 0.");
        return result; // return unchanged
    }

    return result.map(v => v / sum);
}


class Matrix {
    constructor(rows, cols) {
        if (!Number.isInteger(rows) || !Number.isInteger(cols) || rows <= 0 || cols <= 0) {
            throw new Error("Matrix constructor requires positive integer dimensions.");
        }
        this.rows = rows;
        this.cols = cols;
        this.data = Array.from({ length: rows }, () => Array(cols).fill(0));
    }

    copy() {
        let m = new Matrix(this.rows, this.cols);
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                m.data[i][j] = this.data[i][j];
            }
        }
        return m;
    }

    static fromArray(arr) {
        if (!Array.isArray(arr)) throw new TypeError("fromArray expects an array.");
        return new Matrix(arr.length, 1).map((_, i) => arr[i]);
    }

    toArray() {
        return this.data.flat();
    }

    randomize() {
        let limit = Math.sqrt(2 / (this.rows + this.cols)) * 2;
        return this.map(() => (Math.random() * 2 - 1) * limit);
    }

    add(n) {
        if (n instanceof Matrix) {
            if (this.rows !== n.rows || this.cols !== n.cols) {
                throw new Error(`Matrix add error: size mismatch ${this.rows}x${this.cols} vs ${n.rows}x${n.cols}`);
            }
            return this.map((e, i, j) => e + n.data[i][j]);
        } else if (typeof n === "number") {
            return this.map(e => e + n);
        } else {
            throw new TypeError("Matrix.add expects a Matrix or number.");
        }
    }

    multiply(n) {
        if (n instanceof Matrix) {
            if (this.rows !== n.rows || this.cols !== n.cols) {
                throw new Error(`Matrix multiply error: size mismatch ${this.rows}x${this.cols} vs ${n.rows}x${n.cols}`);
            }
            return this.map((e, i, j) => e * n.data[i][j]);
        } else if (typeof n === "number") {
            return this.map(e => e * n);
        } else {
            throw new TypeError("Matrix.multiply expects a Matrix or number.");
        }
    }

    static multiply(a, b) {
        if (!(a instanceof Matrix) || !(b instanceof Matrix)) {
            throw new TypeError("Matrix.multiply expects two Matrices.");
        }
        if (a.cols !== b.rows) {
            throw new Error(`Matrix multiply error: a.cols (${a.cols}) must match b.rows (${b.rows})`);
        }

        let result = new Matrix(a.rows, b.cols);
        for (let i = 0; i < result.rows; i++) {
            for (let j = 0; j < result.cols; j++) {
                let sum = 0;
                for (let k = 0; k < a.cols; k++) {
                    sum += a.data[i][k] * b.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        return result;
    }

    static subtract(a, b) {
        if (!(a instanceof Matrix) || !(b instanceof Matrix)) {
            throw new TypeError("Matrix.subtract expects two Matrices.");
        }
        if (a.rows !== b.rows || a.cols !== b.cols) {
            throw new Error(`Matrix subtract error: size mismatch ${a.rows}x${a.cols} vs ${b.rows}x${b.cols}`);
        }
        return new Matrix(a.rows, a.cols).map((_, i, j) => a.data[i][j] - b.data[i][j]);
    }

    static transpose(matrix) {
        if (!(matrix instanceof Matrix)) {
            throw new TypeError("Matrix.transpose expects a Matrix.");
        }
        return new Matrix(matrix.cols, matrix.rows)
            .map((_, i, j) => matrix.data[j][i]);
    }

    map(func) {
        if (typeof func !== "function") throw new TypeError("Matrix.map expects a function.");
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                let val = this.data[i][j];
                this.data[i][j] = func(val, i, j);
            }
        }
        return this;
    }

    static map(matrix, func) {
        if (!(matrix instanceof Matrix)) throw new TypeError("Matrix.map expects a Matrix.");
        return new Matrix(matrix.rows, matrix.cols)
            .map((_, i, j) => func(matrix.data[i][j], i, j));
    }

    print() {
        console.table(this.data);
        return this;
    }

    serialize() {
        return JSON.stringify(this);
    }

    static deserialize(data) {
        if (typeof data === 'string') data = JSON.parse(data);
        if (!("rows" in data && "cols" in data && "data" in data)) {
            throw new Error("Matrix.deserialize: Invalid data format.");
        }
        let m = new Matrix(data.rows, data.cols);
        m.data = data.data;
        return m;
    }
}
