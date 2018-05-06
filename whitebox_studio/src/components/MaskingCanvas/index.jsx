import React from 'react';
import ReactDOM from 'react-dom';
import saveSvgAsPng from '../../../node_modules/save-svg-as-png/saveSvgAsPng';
import './MaskingCanvas.css';
class DrawCanvas extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      index: 0,
      paths: [[]],
      isDrawing: false,
      top: 0,
      left: 0,
      height: 0,
      width: 0,
      mask: [],
      uri: 'data:image/png;base64,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',
    };
  }

  componentWillMount() {
    this.getMeta(this.state.uri);
  }

  componentDidMount() {
    const node = ReactDOM.findDOMNode(this.refs.canvas);
    const rect = node.getBoundingClientRect();
    const { left, top } = rect;
    this.setState({ top, left });
    node.ondragstart = function () { return false; };

    this.getImage(this.state.index);
  }

  getMeta = (url) => {
    const img = new Image();
    img.addEventListener('load', () => {
      this.setState({ width: img.naturalWidth, height: img.naturalHeight });
      const node = ReactDOM.findDOMNode(this.refs.canvas);
      const rect = node.getBoundingClientRect();
      const { left, top } = rect;
      this.setState({ top, left });
    });
    img.src = url;
  }

  handleMouseDown() {
    if (!this.state.isDrawing) {
      this.setState({
        paths: [].concat(this.state.paths, [[]]),
      });
    }
    this.setState({ isDrawing: true });
  }

  handleMouseMove(e) {
    if (this.state.isDrawing) {
      const x = e.pageX - this.state.left;
      const y = e.pageY - this.state.top;
      const paths = this.state.paths.slice(0);
      const activePath = paths[paths.length - 1];
      activePath.push({ x, y });
      this.setState({ paths });
    }
  }

  handleMouseUp() {
    if (this.state.isDrawing) {
      this.setState({ isDrawing: false });
      const node = ReactDOM.findDOMNode(this.refs.canvas);
      saveSvgAsPng.svgAsPngUri(node, {}, (uri) => {
        this.setState({ uri, paths: [[]] });
      });
    }
  }

  save = () => {
    const node = ReactDOM.findDOMNode(this.refs.canvas);
    // saveSvgAsPng.saveSvgAsPng(node, 'diagram.png');
    saveSvgAsPng.svgAsPngUri(node, {}, (uri) => {
      this.setState({ uri });
      console.log(uri);
      this.getMask();
    });
  }

  getMask = () => {
    const { height, width } = this.state;
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    const img = new Image();
    img.onload = () => {
      ctx.drawImage(img, 0, 0);
      const imageData = ctx.getImageData(0, 0, width, height).data;
      const imageToMat = [];
      console.log(imageData);
      for (let row = 0, count = -1; row < width * height; row++) {
        // TODO use dynamic mask colour
        if (imageData[++count] === 0 && imageData[++count] === 0 && imageData[++count] === 255 && imageData[++count] === 255) {
          imageToMat[row] = 1;
        } else {
          imageToMat[row] = 0;
        }
      }
      console.log(imageToMat.length);
      // TODO fetch post mask here.
    };
    img.src = this.state.uri;
  }

  getImage = (index) => {
    const API = `http://55c9e8ce.ngrok.io/api/v1/annotation/DecoyMNIST/${index}`;
    fetch(API)
      .then(response => response.json())
      .then(data => this.setState({uri: data.data}));

  }

  getNextImage = () => {
    this.getImage(this.state.index + 1);
    this.setState({ index: this.state.index + 1 });
  }

  getPreviousImage = () => {
    this.getImage(this.state.index - 1);
    this.setState({ index: this.state.index - 1 });
  }

  render() {
    const { height, width } = this.state;
    const paths = this.state.paths.map((_points) => {
      let path = '';
      const points = _points.slice(0);
      if (points.length > 0) {
        path = `M ${points[0].x} ${points[0].y}`;
        let p1,
          p2,
          end;
        for (let i = 1; i < points.length - 2; i += 2) {
          p1 = points[i];
          p2 = points[i + 1];
          end = points[i + 2];
          path += ` C ${p1.x} ${p1.y}, ${p2.x} ${p2.y}, ${end.x} ${end.y}`;
        }
      }
      return path;
    }).filter(p => p !== '');
    return (
      <div className="MaskingCanvas">
        <h1>Draw to mask</h1>
        <div className="MaskingCanvas-body">
          <svg
            style={{ border: '0px solid black', cursor: 'crosshair' }}
            width={width*10}
            height={height*10}
            ref="canvas"
            onMouseDown={this.handleMouseDown.bind(this)}
            onMouseUp={this.handleMouseUp.bind(this)}
            onMouseMove={this.handleMouseMove.bind(this)}
          >
            <image x={0} y={0} xlinkHref={this.state.uri} height={height*10} width={width*10} />
            {
              paths.map(path => (<path
                key={path}
                stroke={this.props.attributes.color}
                strokeWidth={this.props.attributes.brushSize}
                d={path}
                fill="none"
              />))
            }
          </svg>
        </div>
        <div className="MaskingCanvas-button-container">
          <button onClick={this.getPreviousImage}>previous</button>
          <button onClick={this.save}>save</button>
          <button onClick={this.getNextImage}>next</button>
        </div>
      </div>
    );
  }
}

export default DrawCanvas;
