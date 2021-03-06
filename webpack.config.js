var path = require('path');

module.exports = {

    entry: path.resolve(__dirname, 'src') + '/app/app.js',
    output: {
        path: path.resolve(__dirname, 'dist') + '/app',
        filename: 'bundle.js',
        publicPath: '/app/'
    },
    module: {
        loaders: [
            {
                test: /\.js$/,
                include: path.resolve(__dirname, 'src'),
                loader: 'babel-loader',
                query: {
                    presets: ['react', 'es2017']
                }
                
            },


            {
                test: /\.css$/,
                loader: 'style-loader!css-loader'
            }
            
        ]
    },
    node: {
        fs: 'empty',
        child_process: 'empty',
      }



};
