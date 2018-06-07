import * as tf from '@tensorflow/tfjs';

// init
let x_vals = [];
let y_vals = [];
let a,b;

// random
const getRandomNumberFromRange = (min, max) =>
  Math.floor(Math.random() * (max - min +1 ) + min);

// loss function
const loss = (pred, labels) => pred.sub(labels).square().mean();

// optimizer & learning rate
const learningRate = 0.01;
const optimizer = tf.train.sgd(learningRate);

// setting up data
function Inputdata(x,y){
  x_vals.push(x);
  y_vals.push(y);
  console.log(x_vals);
  console.log(y_vals);
}

// weight
function weight(){
   a = tf.variable(tf.scalar(getRandomNumberFromRange(0,100)));
   b = tf.variable(tf.scalar(getRandomNumberFromRange(0,100)));
   console.log("original a is: ");
   a.print();
   console.log("original b is: ");
   b.print();
}

// model for y=ax + b
function train() {
  if(x_vals.length > 0){
  const ys = tf.tensor1d(y_vals);
  // optimaze
  optimizer.minimize(() => loss(predict(x_vals), ys));

  console.log("new a is: ");
  a.print();
  console.log("new b is: ");
  b.print();
  }
}

//model
function predict(x){
  const xs = tf.tensor1d(x);
  console.log("the xs is : " + xs.print());
  // y = mx+b
  const ys = xs.mul(a).add(b);
  return ys;
}

weight();

for (let i = 0; i < 10; i++) {
   Inputdata(getRandomNumberFromRange(0,100),getRandomNumberFromRange(0,100));
}

train();



//
