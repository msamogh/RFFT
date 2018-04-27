import React from 'react';
import ReactDOM from 'react-dom';
import saveSvgAsPng from '../../../node_modules/save-svg-as-png/saveSvgAsPng.js';


const _url = "https://upload.wikimedia.org/wikipedia/commons/thumb/6/66/Cephalometric_radiograph.JPG/600px-Cephalometric_radiograph.JPG";
const height = 480;
const width = 600;


class DrawCanvas extends React.Component {
  
  constructor() {
    super();
    this.state = {
      paths: [ [] ],
      isDrawing: false,
      top: 0,
      left: 0,
      simplify: false,
      simplifyHighQuality: true,
      simplifyThreshold: 3,
    };
  }
  
  componentDidMount() {
    const node = ReactDOM.findDOMNode(this.refs.canvas);
    const rect = node.getBoundingClientRect();
    const { left, top } = rect;
    this.setState({ top, left });
  }

    
  handleMouseDown() {
    if (!this.state.isDrawing) {
      this.setState({
        paths: [].concat(this.state.paths, [[]])
      });
    }
    this.setState({ isDrawing: true });
  };
  
  handleMouseMove(e) {
    if (this.state.isDrawing) {
      const x = e.pageX - this.state.left;
      const y = e.pageY - this.state.top;
      const paths = this.state.paths.slice(0);
      const activePath = paths[paths.length - 1];
      activePath.push({ x, y });
      this.setState({ paths });
    }
  };
  
  handleMouseUp() {
    if (this.state.isDrawing) {
      this.setState({ isDrawing: false });
    }
  };
  
  toggleSimplify() {
     this.setState({ simplify: !this.state.simplify });
  }
  
  setThreshold(e) {
     this.setState({ simplifyThreshold: e.target.value });
  }

  convertDataURIToBinary = (dataURI) => {
    var BASE64_MARKER = ';base64,';
    var base64Index = dataURI.indexOf(BASE64_MARKER) + BASE64_MARKER.length;
    var base64 = dataURI.substring(base64Index);
    var raw = window.atob(base64);
    var rawLength = raw.length;
    var array = new Uint8Array(new ArrayBuffer(rawLength));

    for(let i = 0; i < rawLength; i++) {
      array[i] = raw.charCodeAt(i);
    }
    return array;
  }

  save = () => {
    const node = ReactDOM.findDOMNode(this.refs.canvas);
    saveSvgAsPng.saveSvgAsPng(node, "diagram.png");
    saveSvgAsPng.svgAsPngUri(node, {}, (uri) => {
      console.log(uri);

      var canvas = document.createElement("canvas");
      canvas.width = width; // imgElement.offsetWidth;
      canvas.height = height;//imgElement.offsetHeight;

      var ctx = canvas.getContext("2d");
      var imgElement = new Image();
      imgElement.onload = function(){
        ctx.drawImage(imgElement,0,0); // Or at whatever offset you like
      };
      imgElement.src = uri;

      var map = ctx.getImageData(0,0,canvas.width,canvas.height);
      var imdata = map.data;

      var r,g,b,a;

      console.log(imdata.length);
      for(var p = 0, len = imdata.length; p < len; p+=4) {
        r = imdata[p]
        g = imdata[p+1];
        b = imdata[p+2];
        a = imdata[p+3];

        if (p < 100) {
          console.log(p, ':' ,r,g,b,a);
        }

        if ((r === 0) && (g === 0) && (b === 255)) {

            // black  = water
             imdata[p] = 0;
             imdata[p+1] = 0;
             imdata[p+2] = 0;

        } else {

            // white = land
             imdata[p] = 255;
             imdata[p+1] = 255;
             imdata[p+2] = 255;                     
        }                   
      }


      ctx.putImageData(map,0,0);
      imgElement.src = canvas.toDataURL();
      
    });
  }
  
  render() {
    const paths = this.state.paths.map(_points => {
      let path = '';
      let points = _points.slice(0);
      if (points.length > 0) {
        path = `M ${points[0].x} ${points[0].y}`;
        var p1, p2, end;
        for (var i = 1; i < points.length - 2; i += 2) {
          p1 = points[i];
          p2 = points[i + 1];
          end = points[i + 2];
          path += ` C ${p1.x} ${p1.y}, ${p2.x} ${p2.y}, ${end.x} ${end.y}`;
        }
      }
      return path;
    }).filter(p => p !== '');
    return (
      <div>
        <h1>Draw to mask</h1>
        <label>
          <input
            type="number"
            value={this.state.simplifyThreshold}
            onChange={this.setThreshold.bind(this)}
          />
        </label>
        <button onClick={this.save}>save</button>
        <br />
        <svg
          style={{ border: '0px solid black', cursor: 'crosshair' }}
          width={width}
          height={height}
          ref="canvas"
          onMouseDown={this.handleMouseDown.bind(this)}
          onMouseUp={this.handleMouseUp.bind(this)}
          onMouseMove={this.handleMouseMove.bind(this)}
         >
          <image x={0} y={0} xlinkHref={_url} height={height} width={width} />
          {
            paths.map(path => {
              return (<path
                key={path}
                stroke="blue"
                strokeWidth={this.state.simplifyThreshold}
                d={path}
                fill="none"
              />);
            })
          }
        </svg>
      </div>
    );
  }
}

export default DrawCanvas;