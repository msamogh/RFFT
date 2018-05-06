import React from 'react';
import TopBar from '../TopBar';
import MaskingCanvas from '../MaskingCanvas';
import Home from '../Home';
import ExperimentSpace from '../ExperimentSpace';
import './App.css';

class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      page: 'home',
      currentExperiment: {},
    };
  }

  goToWorkspace = (page) => (currentExperiment) => {
    this.setState({ page, currentExperiment });
  }

  onNavigationClick = (page) => {
    this.setState({ page });
  }

  renderBody = () => {
    switch (this.state.page) {
      case 'home': return (<Home goToWorkspace={this.goToWorkspace('experimentSpace')} />);
      case 'textAnotator': return (<div>textAnotator</div>);
      case 'imageMasker': return (<MaskingCanvas />);
      case 'experimentSpace': return (<ExperimentSpace>workSpace </ExperimentSpace>);
      default: return (<div>home</div>);
    }
  }

  render() {
    return (
      <div className="App">
        <TopBar goHome={()=>this.onNavigationClick('home')}/>
        {this.renderBody()}
      </div>
    );
  }
}

export default App;
