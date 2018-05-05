import React from 'react';
import Annotator from '../../MaskingCanvas';
import './WorkSpace.css';

class WorkSpace extends React.Component {
  renderAction = () => {
    switch (this.props.state) {
      case 'Annotate' : return (<Annotator/>);
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
