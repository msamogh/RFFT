import React from 'react';
import SideBar from './NavigationBar';
import WorkSpace from './WorkSpace';
import AttributesBar from './AttributesBar';
import './ExperimentSpace.css';



class ExperimentSpace extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      experiment: {},
      state: 'home',
      attributes: {
        color: '#0000ff',
        brushSize: 50,
      }
    };
  }

  navigateToAction = (state) => () => {
    this.setState({state});
  }

  onAttributesChange = (attributes) => {
    this.setState({attributes: {...this.state.attributes, ...attributes}})
  }

  render() {
    const {state, attributes} = this.state;
    return (
      <div className="ExperimentSpace">
        <SideBar onClick={this.navigateToAction}/>
        <WorkSpace state={state} attributes={attributes}/>
        <AttributesBar state={state} attributes={attributes} onAttributesChange={this.onAttributesChange}/>
      </div>
    );
  }
}

export default ExperimentSpace;
