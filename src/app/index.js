// tensorflow
import * as tf from '@tensorflow/tfjs';

//react
import React, { Component } from 'react';
var ReactDOM = require('react-dom');
import { Router, Route, browserHistory, Link} from 'react-router';


// other tf files
import {ControllerDataset} from './data_set';
import {Webcam} from './webcam';
//import {Colony} from './colony'

//ui
import { Layout , Divider, Button, Menu} from 'antd';
const { Header, Footer, Sider, Content } = Layout;
import 'antd/dist/antd.min.css';
require('./style.css');


// Ethereum
const contractAddress = '0x9c4795922ac0e56d5013a777046108a4d751b382';
const abi = require('../../Contract/abi');
const mycontract = web3.eth.contract(abi);
const myContractInstance = mycontract.at(contractAddress);

// ipfs
var ipfsAPI = require('ipfs-api');
var ipfs = ipfsAPI('ipfs.infura.io', '5001', {protocol: 'https'});

//files
const fileDownload = require('js-file-download');


//smile or not smile
const NUM_CLASSES = 2;

// A webcam class that generates Tensors from the images from the webcam.
const webcam = new Webcam(document.getElementById('webcam'));

// The dataset object where we will store activations.
const controllerDataset = new ControllerDataset(NUM_CLASSES);

let mobilenet;
let model;

let counter = {"smile":0, "not smile":0};
const result = {0:"smile", 1:"not smile"};
// Loads mobilenet and returns a model that returns the internal activation
// we'll use as input to our classifier model.
async function loadMobilenet() {
  const mobilenet = await tf.loadModel(
      'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');

  // Return a model that outputs an internal activation.
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  console.log("got mobilenet")
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
}


// screenshots of samples
  const ExampleHandler = (label) => {
  tf.tidy( () => {
    const img = webcam.capture();
    controllerDataset.addExample(mobilenet.predict(img), label);
    console.log("added")
    // Draw the preview thumbnail.
    //ui.drawThumb(img, label);
  });
};



async function issueToken(){
  var getData = myContractInstance.sendToken.getData(web3.eth.accounts[0], 2);
  await web3.eth.sendTransaction({from:web3.eth.accounts[0], to:contractAddress, data:getData},(err,res) =>{
    console.log("tokenIssued");
  });
}
/**
 * Sets up and trains the classifier.
 */
async function train(dense_unit) {
  if (controllerDataset.xs == null) {
    throw new Error('Add some examples before training!');
    console.log("no data");
  }

  counter = {"smile":0, "not smile":0}


  // Creates a 2-layer fully connected model. By creating a separate model,
  // rather than adding layers to the mobilenet model, we "freeze" the weights
  // of the mobilenet model, and only train weights from the new model.
  model = tf.sequential({
    layers: [
      // Flattens the input to a vector so we can use it in a dense layer. While
      // technically a layer, this only performs a reshape (and has no training
      // parameters).
      tf.layers.flatten({inputShape: [7, 7, 256]}),
      // Layer 1
      tf.layers.dense({
        units: dense_unit,
        activation: 'relu',
        kernelInitializer: 'varianceScaling',
        useBias: true
      }),
      // Layer 2. The number of units of the last layer should correspond
      // to the number of classes we want to predict.
      tf.layers.dense({
        units: NUM_CLASSES,
        kernelInitializer: 'varianceScaling',
        useBias: false,
        activation: 'softmax'
      })
    ]
  });

  // Creates the optimizers which drives training of the model.
  const optimizer = tf.train.adam(0.0001);
  // We use categoricalCrossentropy which is the loss function we use for
  // categorical classification which measures the error between our predicted
  // probability distribution over classes (probability that an input is of each
  // class), versus the label (100% probability in the true class)>
  model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});
  // We parameterize batch size as a fraction of the entire dataset because the
  // number of examples that are collected depends on how many examples the user
  // collects. This allows us to have a flexible batch size.
  const batchSize =
      Math.floor(controllerDataset.xs.shape[0] * 0.4);
  if (!(batchSize > 0)) {
    throw new Error(
        `Batch size is 0 or NaN. Please choose a non-zero fraction.`);
  }
  // Train the model! Model.fit() will shuffle xs & ys so we don't have to.
  model.fit(controllerDataset.xs, controllerDataset.ys, {
    batchSize,
    epochs: 20,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        console.log('Loss: ' + logs.loss.toFixed(5));
        await tf.nextFrame();
      }
    }
  });
  console.log("training")
}

let isPredicting = false;

async function predict() {
  console.log(isPredicting)
  // model = tf.sequential({
  //   layers: [
  //     // Flattens the input to a vector so we can use it in a dense layer. While
  //     // technically a layer, this only performs a reshape (and has no training
  //     // parameters).
  //     tf.layers.flatten({inputShape: [7, 7, 256]}),
  //     // Layer 1
  //     tf.layers.dense({
  //       units: 10,
  //       activation: 'relu',
  //       kernelInitializer: 'varianceScaling',
  //       useBias: true
  //     }),
  //     // Layer 2. The number of units of the last layer should correspond
  //     // to the number of classes we want to predict.
  //     tf.layers.dense({
  //       units: NUM_CLASSES,
  //       kernelInitializer: 'varianceScaling',
  //       useBias: false,
  //       activation: 'softmax'
  //     })
  //   ]
  // });

  while (isPredicting) {
    const predictedClass = tf.tidy(() => {
      // Capture the frame from the webcam.
      const img = webcam.capture();
      // Make a prediction through mobilenet, getting the internal activation of
      // the mobilenet model.
      const activation = mobilenet.predict(img);
      // Make a prediction through our newly-trained model using the activation
      // from mobilenet as input.

      const predictions = model.predict(activation);
      // Returns the index with the maximum probability. This number corresponds
      // to the class the model thinks is the most probable given the input.
      return predictions.as1D().argMax();

    });
    const classId = (await predictedClass.data())[0];
    console.log(result[classId]);
    counter[result[classId]] +=1;
    if (counter["smile"] >= 100){
      issueToken();
      break;
    }
    predictedClass.dispose();
    await tf.nextFrame();

  }
  alert("congrats, you smiled");
}

 const onTrain = async () =>  {
  console.log('Training...');
  await tf.nextFrame();
  await tf.nextFrame();
  isPredicting = false;
  train();
};


const onPredict = () => {
  console.log("predicting");
  isPredicting = true;
  predict();
};




async function init() {
  try {
    await webcam.setup();
  } catch (e) {
    console.log("nowebcam");
  }
  mobilenet = await loadMobilenet();
  tf.tidy(() => mobilenet.predict(webcam.capture()));

  console.log("camera is on")
}

window.addEventListener('load', function() {
        // Checking if Web3 has been injected by the browser (Mist/MetaMask)
           if (typeof web3 !== 'undefined') {
                // Use Mist/MetaMask's provider
                window.web3 = new Web3(web3.currentProvider);
            } else {
                console.log('No web3? You should consider trying MetaMask!')
                // fallback - use your fallback strategy (local node / hosted node + in-dapp id mgmt / fail)
                window.web3 = new Web3(new Web3.providers.HttpProvider("https://localhost:8545"));
        }
      });


class App extends Component{
  render() {
    return(
      <Router history={browserHistory}>
        <Route path={"/"} component={MLapp}></Route>
      </Router>
    );
  }
};


class MLapp extends Component {

  constructor(props) {
    super(props);

    this.state = {
    smileLabel:0,
    account:web3.eth.accounts[0],
    others:0,
    smilecoin:0,
    added_file_hash: null

  }
  this.captureFile = this.captureFile.bind(this)
  this.handleSubmit = this.handleSubmit.bind(this)
  this.loadModel = this.loadModel.bind(this);
  this.saveModel = this.saveModel.bind(this);
  this.startPredict = this.startPredict.bind(this);
  this.startTrain = this.startTrain.bind(this);
  this.incrementSmile = this.incrementSmile.bind(this);
  this.incrementOther = this.incrementOther.bind(this);
  this.showAccount = this.showAccount.bind(this);

}

  async startTrain(){
    await train(this.state.smileLabel);
    console.log("trained");
  }

  async startPredict(){
    await onPredict();
    console.log("done!");
  }

  incrementSmile() {
    ExampleHandler(0);
    this.setState({ smileLabel: this.state.smileLabel + 1 });
  }

  incrementOther() {
    ExampleHandler(1);
    //alert(this.state.account);
    this.setState({ others: this.state.others + 1 });
  }

  async showAccount(){
    await myContractInstance.getBalance(web3.eth.accounts[0],function(err,result){
      this.setState( {smilecoin:result.c[0]});
    }.bind(this));
    this.setState( {account:web3.eth.accounts[0]});
  }

  async saveModel(){
    const saveResult = await model.save(tf.io.browserHTTPRequest(
      'http://localhost:8080/ipfs/upload',
      {method: 'POST', headers: {'Content-Type': 'application/json', 'key':'tfmodel'}}));
  }


  async loadModel(){
    //const model = await tf.loadModel('https://ipfs.infura.io/ipfs/QmeFJwAHQpTAd8Ni2iWWqZ4jiWLPFxveHsZbe5gCsFf8VL');
    //alert("loaded!");


    const modelCID = 'QmeFJwAHQpTAd8Ni2iWWqZ4jiWLPFxveHsZbe5gCsFf8VL'
    const weightsCID = 'QmXqNrKne253rkqi86GS6qKDGfBQFkVVuWf3ywQYVUddSh'

    var filesModel = await ipfs.files.get(modelCID);
    //console.log(filesModel[0].content);
    fileDownload(filesModel[0].content.toString('utf8'), 'smileModel.json');

    var filesWeights = await ipfs.files.get(weightsCID);
    //console.log(filesWeights);

    fileDownload(filesWeights[0].content.toString('utf8'), 'smileModel.weights.bin');

    var modelurl = "https://ipfs.infura.io/ipfs/QmeFJwAHQpTAd8Ni2iWWqZ4jiWLPFxveHsZbe5gCsFf8VL";
    var weightsurl = "https://ipfs.infura.io/ipfs/QmXqNrKne253rkqi86GS6qKDGfBQFkVVuWf3ywQYVUddSh";
 
    
  }

  componentDidMount() {
     this.setState( {account:web3.eth.accounts[0]});
  }

  componentWillMount(){
    this.setState( {account:web3.eth.accounts[0]});
  }
  
  captureFile (event) {
  event.stopPropagation()
  event.preventDefault()
  const file = event.target.files[0]
  let reader = new window.FileReader()
  reader.onloadend = () => this.saveToIpfs(reader)
  reader.readAsArrayBuffer(file)
}

// get and load the model 
async captureModel (event) {
  event.stopPropagation()
  event.preventDefault()
  const modelFile = await event.target.files[0];
  //const weightsFile = await event.target.files[1];
  const model = await tf.loadModel(tf.io.browserFiles([modelFile, weightsFile]));
}

saveToIpfs (reader) {
  let ipfsId
  const buffer = Buffer.from(reader.result)
  ipfs.add(buffer, { progress: (prog) => console.log(`received: ${prog}`) })
    .then((response) => {
      console.log(response)
      ipfsId = response[0].hash
      console.log(ipfsId)
      this.setState({added_file_hash: ipfsId})
    }).catch((err) => {
      console.error(err)
    })
}

handleSubmit (event) {
  event.preventDefault()
}

  render() {
    return (
      <Layout>
        <Header>Smile Coin</Header>
          <Divider>Smile Coin</Divider>
          <p>learning rate: 0.0001 Batch size: 0.4 Epochs: 20 Hidden units: 100</p>
          <Content>
          <p>the account number is: {this.state.account}</p>
          <p>You have : {this.state.smilecoin} smile coins now</p>
          <Button onClick={this.startTrain}>Train</Button>
          <Button type="primary" onClick={this.startPredict}>Go</Button>
          <Divider>Add Samples </Divider>
          <Button  onClick={this.incrementSmile}>simileSample:{this.state.smileLabel}</Button>
          <Button  onClick={this.incrementOther}>otherSample:{this.state.others}</Button>
          <br/>
          <br/>
          <Button type="primary" onClick={this.showAccount}>Show Account</Button>
          <Button type="primary" onClick={this.saveModel}>Save Model</Button>
          <Button type="primary" onClick={this.loadModel}>Load Model</Button>
          </Content>
          <div>
          <form id='captureMedia' onSubmit={this.handleSubmit}>
            <input type='file' onChange={this.captureFile} />
          </form>
          <p>Model below</p>
          <form id='captureMedia' onSubmit={this.handleSubmit}>
            <input type='file' onChange={this.captureModel}/>
          </form>
          <div>
            <a target='_blank'
              href={'https://ipfs.infura.io/ipfs/' + this.state.added_file_hash}>
              {this.state.added_file_hash}
            </a>
          </div>
        </div>
      </Layout>
    );
  }
}
init();

export default App;
