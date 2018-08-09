var download = require('download-file')
 
var url = "https://ipfs.infura.io/ipfs/QmeFJwAHQpTAd8Ni2iWWqZ4jiWLPFxveHsZbe5gCsFf8VL"
 
var options = {
    directory: "./app/",
    filename: "cat.bin"
}
 
download(url, options, function(err){
    if (err) throw err
    console.log("meow")
}) 