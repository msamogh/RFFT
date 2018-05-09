import React from 'react';
import constants from '../../../constants';
import './AttributesBar.css';

class AttributesBar extends React.Component {

  constructor(props) {
    super(props);
    this.state = {
      trainAtrributes: {
        useAnnotations: false,
        numberOfAnnotations: 0,
        numberOfEpochs: 0,
        hypothesisWeight: 0,
      },
    };
  }

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
        <label htmlFor="brushColor">Brush Color </label>
        <input id="brushColor" type="color" value={this.props.attributes.color} onChange={this.annotatorColorChange}/>
      </div>
      <label htmlFor="brushSize">{`Brush Size ${this.props.attributes.brushSize}`}</label>
      <input id="brushSize" type="range" min="1" max="10" value={this.props.attributes.brushSize} step="1" onChange={this.annotatorBrushSizeChange}/>
    </div>
  )

  train = () => {
    const API = `${constants.API}/train/${this.props.experiment.id}`;
    const trainAtrributes = {
      num_annotations: this.state.trainAtrributes.useAnnotations ? this.state.trainAtrributes.numberOfAnnotations : 0,
      num_epochs: this.state.trainAtrributes.numberOfEpochs,
      hypothesis_weight: this.state.trainAtrributes.hypothesisWeight,
    }
    fetch(API, {method: 'POST', body: JSON.stringify(trainAtrributes)})
      .then(response => response.json())
      .then(data => {console.log(data)});
    console.log(trainAtrributes);

  }

  trainParamChange = (param) => (event) => {
    this.setState({trainAtrributes: {...this.state.trainAtrributes, [param]: event.target.value}});
  }

  handleClick = (event) => {
    this.setState({trainAtrributes: {...this.state.trainAtrributes, useAnnotations: event.target.checked}});
  }

  trainAttributes = () => (
    <div className="train-atrributes">
      <div>
        <label htmlFor="useAnnotations">Use Annotations while training</label><br />
        <input type="checkbox" id="useAnnotations" onChange={this.handleClick}/>
      </div>

      <label htmlFor="numberOfAnnotations">Number of Annotations to use</label>
      <input type="number" id="numberOfAnnotations" max="30" onChange={this.trainParamChange('numberOfAnnotations')}/>
      
      <label htmlFor="numberOfEpochs">Number of Epochs</label>
      <input type="number" id="numberOfEpochs" onChange={this.trainParamChange('numberOfEpochs')}/>

      <label htmlFor="hypothesisWeight">Hypothesis weight</label>
      <input type="number" id="hypothesisWeight" min="0" max="100000" step="10" onChange={this.trainParamChange('hypothesisWeight')}/>

      <button onClick={this.train}>TRAIN</button>
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
