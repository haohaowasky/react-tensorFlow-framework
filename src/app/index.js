// tensorflow
import * as tf from '@tensorflow/tfjs';

//react
import React, { Component } from 'react';
var ReactDOM = require('react-dom');

// other tf files
import {ControllerDataset} from './data_set';
import {Webcam} from './webcam';

//ui
import { Layout , Divider, Button} from 'antd';
const { Header, Footer, Sider, Content } = Layout;
import 'antd/dist/antd.min.css';
require('./style.css');

// Ethereum
const contractAddress = '0x9c4795922ac0e56d5013a777046108a4d751b382';
const abi = require('../../Contract/abi');
const mycontract = web3.eth.contract(abi);
const myContractInstance = mycontract.at(contractAddress);

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
  var getData = myContractInstance.sendToken.getData(web3.eth.accounts[0], 200);
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
  console.log("predicting")
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


class MLapp extends Component {

  constructor(props) {
    super(props);

    this.state = {
    smileLabel:0,
    account:web3.eth.accounts[0],
    others:0,
    smilecoin:0
  }

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
    //this.setState( {account:web3.eth.accounts[0]});
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

  componentDidMount() {
     this.setState( {account:web3.eth.accounts[0]});
  }

  componentWillMount(){
    this.setState( {account:web3.eth.accounts[0]});
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
          </Content>
      </Layout>
    );
  }
}
init();

export default MLapp;
