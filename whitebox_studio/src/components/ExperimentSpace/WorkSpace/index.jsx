import React from 'react';
import Annotator from '../../MaskingCanvas';
import LossGraph from '../../LossGraph';
import './WorkSpace.css';

class WorkSpace extends React.Component {
  renderAction = () => {
    switch (this.props.state) {
      case 'Annotate' : return (<Annotator attributes={this.props.attributes}/>);
      case 'Train' : return (<LossGraph />)
      default: return('home');
    }
  }

  render() {
    return (
      <div className="WorkSpace">
        {this.renderAction()}   
      </div>
    );
  }
}

export default WorkSpace;
