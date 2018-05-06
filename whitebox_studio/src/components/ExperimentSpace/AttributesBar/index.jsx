import React from 'react';
import './AttributesBar.css';

class AttributesBar extends React.Component {

  annotatorColorChange = (event) => {
    const atrributes = {
      color: event.target.value,
    }
    this.props.onAttributesChange(atrributes);
  }

  annotatorBrushSizeChange = (event) => {
    const atrributes = {
      brushSize: event.target.value,
    }
    this.props.onAttributesChange(atrributes);
  }

  annotatorAttributes = () => (
    <div className="annotator-atrributes">
      <h2>{this.props.attributes.brushSize}</h2>
      <input type="range" min="10" max="100" value={this.props.attributes.brushSize} step="10" onChange={this.annotatorBrushSizeChange}/>
      <input type="color" value={this.props.attributes.color} onChange={this.annotatorColorChange}/>
    </div>
  )

  renderStateAttributes = () => {
    switch (this.props.state) {
      case 'Annotate' : return (this.annotatorAttributes());
      default: return('home');
    }
  }

  render() {
    return (
      <div className="AttributesBar">
        {this.renderStateAttributes()}   
      </div>
    );
  }
}

export default AttributesBar;
