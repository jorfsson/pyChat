const fs = require('fs')
fs.readFile('./data/smallerData.txt', (err, data) => {
  if (err) throw err
  fs.writeFileSync('./data/smallerData.txt', data.toString().slice(0, 10000))
})
