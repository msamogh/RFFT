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
      <div>
        <label htmlFor="brushColor">Brush Color: </label>
        <input id="brushColor" type="color" value={this.props.attributes.color} onChange={this.annotatorColorChange}/>
      </div>
      <label htmlFor="brushSize">{`Brush Size: ${this.props.attributes.brushSize}`}</label>
      <input id="brushSize" type="range" min="1" max="10" value={this.props.attributes.brushSize} step="1" onChange={this.annotatorBrushSizeChange}/>
    </div>
  )

  trainAttributes = () => (
    <div className="train-atrributes">
      <div>
        <label htmlFor="useAnnotations">Use Annotations:</label>
        <input type="checkbox" id="useAnnotations"/>
      </div>
      <label htmlFor="numberOfAnnotations">Number of Annotations:</label>
      <input type="number" id="numberOfAnnotations" placeholder="number of annotations" defaultValue={10}/>
      <label htmlFor="numberOfEpochs">Number of Epochs: </label>
      <input type="number" id="numberOfEpochs" placeholder="number of epochs" defaultValue={10}/>
    </div>
  )

  renderStateAttributes = () => {
    switch (this.props.state) {
      case 'Annotate' : return (this.annotatorAttributes());
      case 'Train' : return (this.trainAttributes());
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
