import * as tf from '@tensorflow/tfjs';

// const y = tf.tidy(() => {
//    const one = tf.scalar(1);
//    const a = tf.scalar(2);
//
//    // b will not be cleaned up by the tidy. a and one will be cleaned up
//    // when the tidy ends.
//    b = tf.keep(a.square());
//
//    console.log('numTensors (in tidy): ' + tf.memory().numTensors);
//
//    // The value returned inside the tidy function will return
//    // through the tidy, in this case to the variable y.
//    return b.add(one);
// });

//
// const values = [];
// for (var i = 0; i < 30; i++){
//   values[i] = Math.random()*(100 - 1) + 1;
// }
//
// const shape = [2,5,3];
//
// const data = tf.tensor(values, shape);
//
// console.log(data.toString());
//



// y = ax3 + bx2 + cx + d. the weight
const a = tf.variable(tf.scalar(Math.random()));
const b = tf.variable(tf.scalar(Math.random()));
const c = tf.variable(tf.scalar(Math.random()));
const d = tf.variable(tf.scalar(Math.random()));

const x = tf.variable(tf.scalar(Math.random()));

// a.print();
// b.print();
// c.print();
// d.print();

// predict function the model 
function predict(x) {
  // y = a * x ^ 3 + b * x ^ 2 + c * x + d
  return tf.tidy(() => {
    return a.mul(x.pow(tf.scalar(3))) // a * x^3
      .add(b.mul(x.square())) // + b * x ^ 2
      .add(c.mul(x)) // + c * x
      .add(d); // + d
  });
}

predict(x).print();

// loss functions
function loss(predictions, labels) {
  // Subtract our labels (actual values) from predictions, square the results,
  // and take the mean.
  const meanSquareError = predictions.sub(labels).square().mean();
  return meanSquareError;
}


// optimizer

const learningRate = 0.5;
const optimizer = tf.train.sgd(learningRate);



function train(xs, ys, numIterations = 75) {

  const learningRate = 0.5;
  const optimizer = tf.train.sgd(learningRate);

  for (let iter = 0; iter < numIterations; iter++) {
    optimizer.minimize(() => {
      const predsYs = predict(xs);
      return loss(predsYs, ys);
    });
  }
}
