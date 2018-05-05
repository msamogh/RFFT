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
    };
  }

  navigateToAction = (state) => () => {
    this.setState({state});
  }

  render() {
    const {state} = this.state;
    return (
      <div className="ExperimentSpace">
        <SideBar onClick={this.navigateToAction}/>
        <WorkSpace state={state}/>
        <AttributesBar state={state}/>
      </div>
    );
  }
}

export default ExperimentSpace;
