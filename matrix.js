function normaliseMatrix(a) {
    let result = new Matrix(a.rows, a.cols);
    let sum = 0;
    for (let i = 0; i < a.rows; i++) {
        for (let j = 0; j < a.cols; j++) {
            result.data[i][j] = a.data[i][j];
            sum += Math.abs(result.data[i][j]);
        }
    }
    for (let i = 0; i < a.rows; i++) {
        for (let j = 0; j < a.cols; j++) {
            result.data[i][j] /= sum;
        }
    }
    return result;
}


class Matrix {
    constructor(rows, cols) {
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
        return new Matrix(arr.length, 1).map((e, i) => arr[i]);
    }

    toArray() {
        let arr = [];
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                arr.push(this.data[i][j]);
            }
        }
        return arr;
    }

    randomize() {
        let limit = Math.sqrt(2 / (this.rows + this.cols)) * 2;
        return this.map(() => (Math.random() * 2 - 1) * limit);
    }

    add(n) {
        if (n instanceof Matrix) {
            if (this.rows !== n.rows || this.cols !== n.cols) {
                console.error('Matrix add error: size mismatch');
                return;
            }
            return this.map((e, i, j) => e + n.data[i][j]);
        } else {
            return this.map(e => e + n);
        }
    }

    multiply(n) {
        if (n instanceof Matrix) {
            if (this.rows !== n.rows || this.cols !== n.cols) {
                console.error('Matrix multiply error: size mismatch');
                return;
            }
            return this.map((e, i, j) => e * n.data[i][j]);
        } else {
            return this.map(e => e * n);
        }
    }

    static multiply(a, b) {
        if (a.cols !== b.rows) {
            console.error('Matrix multiply error: a.cols must match b.rows');
            return undefined;
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
        if (a.rows !== b.rows || a.cols !== b.cols) {
            console.error('Matrix subtract error: size mismatch');
            return;
        }
        return new Matrix(a.rows, a.cols).map((_, i, j) => a.data[i][j] - b.data[i][j]);
    }

    static transpose(matrix) {
        return new Matrix(matrix.cols, matrix.rows)
            .map((_, i, j) => matrix.data[j][i]);
    }

    map(func) {
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                let val = this.data[i][j];
                this.data[i][j] = func(val, i, j);
            }
        }
        return this;
    }

    static map(matrix, func) {
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
        if (typeof data === 'string') {
            data = JSON.parse(data);
        }
        let m = new Matrix(data.rows, data.cols);
        m.data = data.data;
        return m;
    }
}