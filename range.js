var Range = function(start, end, step) {
  this.start = start
  this.end = end
  this.step = step
  this.backwards = this.end < this.start ? true : false
};

Range.prototype.size = function () {
  let size = 0
  if (this.step === undefined) {
    this.step = 1
  }
  if (this.end === undefined) {
    return 1
  }
  if (this.backwards) {
    for (let i = this.start; i >= this.end; i -= this.step) {
      size++
    }
  } else {
    for (let i = this.start; i <= this.end; i += this.step){
      size++
    }
  }
  return size
};

Range.prototype.each = function (callback) {
  if (this.step === undefined) {
    this.step = 1
  }
  if (this.end === undefined) {
    return callback(this.start)
  }
  if (this.backwards) {
    for (let i = this.start; i >= this.end; i -= this.step) {
      callback(i)
    }
  } else {
    for (let i = this.start; i <= this.end; i += this.step){
      callback(i)
    }
  }
};

Range.prototype.includes = function (val) {
  let result = [];
  this.each((num)=>{result.push(num === val)})
  return result.reduce((x, y)=> y === true ? true : x, false)
};

var countdown = new Range(10, 0, -2); // Let's count down by twos
    var elements = [];
    countdown.each(function(val){elements.push(val);});

console.log(elements)
