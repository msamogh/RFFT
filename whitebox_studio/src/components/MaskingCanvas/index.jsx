import React from 'react';
import ReactDOM from 'react-dom';
import constants from '../../constants';
import './MaskingCanvas.css';
class DrawCanvas extends React.Component {
  constructor(props) {
    super(props);
    const mask = new Array(28*28);
    mask.fill(0,0,28*28);
    this.state = {
      index: 0,
      isDrawing: false,
      top: 0,
      left: 0,
      height: 0,
      width: 0,
      mask,
      multiplier: 10,
      uri: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAAAAABXZoBIAAAABGdBTUEAALGPC/xhBQAAAAFzUkdCAK7OHOkAAAAgY0hSTQAAeiYAAICEAAD6AAAAgOgAAHUwAADqYAAAOpgAABdwnLpRPAAAAAlwSFlzAAAOwwAADsMBx2+oZAAAAAl2cEFnAAAAHAAAABwAh7EitAAAANBJREFUKM9jYBjMgFlISKiuY73Usv/f61Ek5FTiZq36CwIP1/z9dNgBWc7w3V8o+B0bFGShjqJR6DZY5ti27x+x2BcwJ/vv37PcDNqzsLmGj3HW3yjcju3+u48JpyT3vr9uuLUqf3y4IIcRl2zgh79/yyVxyeru+vt3mjQuWYHYP39347b459+fDthl9Jq2//17HquH1Kc8BQbhr21YpCSK7oKC96QfppS401VwyAdimim0GhwrhwM4MaTM1zwCSX1p5cZiWwdQ5kp7iwADfQAA+21rhoBnKQwAAAAldEVYdGRhdGU6Y3JlYXRlADIwMTgtMDItMTlUMTI6NTg6NDkrMDk6MDAeWgaLAAAAJXRFWHRkYXRlOm1vZGlmeQAyMDE4LTAyLTE5VDEyOjU4OjQ5KzA5OjAwbwe+NwAAABd0RVh0cG5nOmJpdC1kZXB0aC13cml0dGVuAAinxCzyAAAAAElFTkSuQmCC',
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
      // alert(img.naturalWidth, img.naturalHeight);
      const node = ReactDOM.findDOMNode(this.refs.canvas);
      const rect = node.getBoundingClientRect();
      const { left, top } = rect;
      this.setState({ top, left });
    });
    img.src = url;
  }

  handleMouseDown() {
    this.setState({ isDrawing: true });
  }

  handleMouseMove(e) {
    if (this.state.isDrawing) {
      const x = parseInt((e.pageX - this.state.left)/10, 10);
      const y = parseInt((e.pageY - this.state.top)/10, 10);

      const brushSize = this.props.attributes.brushSize - 1;
  

      const x1 = Math.max(x - brushSize,0);
      const y1 = Math.max(y- brushSize, 0);
      const x2 = Math.min(x + brushSize, 27);
      const y2 = Math.min(y + brushSize, 27);

      let mask = [...this.state.mask];

      for (let q = y1; q <= y2; q++) {
        for (let p = x1; p <= x2; p++) {
          mask[q*28 + p] = 1;
        }
      }

      this.setState({mask});
    }
  }

  handleMouseUp() {
    if (this.state.isDrawing) {
      this.setState({ isDrawing: false });
    }
  }

  save = () => {
    const {index, mask} = this.state;
    const API = `${constants.API}/annotation/DecoyMNIST/${index}`;
    let body = {mask};
    body = JSON.stringify(body);
    fetch(API, { method: 'PUT', body})
  }
  clearMask = () => {
    const mask = new Array(28*28);
    mask.fill(0,0,28*28);
    this.setState({mask});
  }

  getImage = (index) => {
    const API = `${constants.API}/annotation/DecoyMNIST/${index}`;
    fetch(API)
      .then(response => response.json())
      .then(data => {
        let mask = data.mask;
        if (mask == null) {
          mask = new Array(28*28);
          mask.fill(0,0,28*28);
        } else if (mask.length < 1) {
          mask = new Array(28*28);
          mask.fill(0,0,28*28);
        }
        this.setState({uri: data.data, mask});
        
      });

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
    const { height, width, multiplier } = this.state;
    return (
      <div className="MaskingCanvas">
        <h1>Draw to mask</h1>
        <div className="MaskingCanvas-body">
          <svg
            style={{ cursor: 'crosshair' }}
            width={width * multiplier}
            height={height * multiplier}
            ref="canvas"
            onMouseDown={this.handleMouseDown.bind(this)}
            onMouseUp={this.handleMouseUp.bind(this)}
            onMouseMove={this.handleMouseMove.bind(this)}
          >
            <image x={0} y={0} xlinkHref={this.state.uri} height={height * multiplier} width={width * multiplier} />
             {
              this.state.mask.map((pixel, index) => pixel ? (<rect
                x={(index%width)*multiplier}
                y={parseInt(index/width, 10)*multiplier}
                width={multiplier}
                height={multiplier}
                key={index}
                stroke={this.props.attributes.color}
                fill={this.props.attributes.color}
              />): null)
            }
          </svg>
        </div>
        <div className="MaskingCanvas-button-container">
          <button onClick={this.getPreviousImage}>previous</button>
          <button onClick={this.save}>save</button>
          <button onClick={this.clearMask}>clear</button>
          <button onClick={this.getNextImage}>next</button>
        </div>
      </div>
    );
  }
}

export default DrawCanvas;
