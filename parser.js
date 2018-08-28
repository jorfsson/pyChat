const fs = require('fs')
const comments = fs.readFileSync('./data/comments.txt').toString()

let newComments = comments.replace(/[^a-z,.!?']/, "")

fs.writeFileSync('data.txt', newComments)
