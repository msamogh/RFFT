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
        <label for="brushColor">Brush Color</label>
        <input id="brushColor" type="color" value={this.props.attributes.color} onChange={this.annotatorColorChange}/>
      </div>
      <label for="brushSize">{`Brush Size${this.props.attributes.brushSize}`}</label>
      <input id="brushSize" type="range" min="10" max="100" value={this.props.attributes.brushSize} step="10" onChange={this.annotatorBrushSizeChange}/>
    </div>
  )

  trainAttributes = () => (
    <div className="train-atrributes">
      <div>
        <label for="useAnnotations">Use Annotations while training</label><br />
        <input type="checkbox" id="useAnnotations"/>
      </div>

      <label for="numberOfAnnotations">Number of Annotations to use</label>
      <input type="number" id="numberOfAnnotations" max="30"/>
      
      <label for="numberOfEpochs">Number of Epochs</label>
      <input type="number" id="numberOfEpochs" />

      <label for="numberOfEpochs">Hypothesis weight</label>
      <input type="number" id="hypothesisWeight" min="0" max="100000" step="10"/>
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
        <h4>Properties</h4>
        {this.renderStateAttributes()}   
      </div>
    );
  }
}

export default AttributesBar;
