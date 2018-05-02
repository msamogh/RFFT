import React from 'react';
import NavigationBar from '../NavigationBar';
import MaskingCanvas from '../MaskingCanvas';
import Home from '../Home';
import './App.css';

class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      page: 'home',
    };
  }

  onNavigationClick = (page) => {
    this.setState({ page });
  }

  renderBody = () => {
    switch (this.state.page) {
      case 'home': return (<Home navigateToMaskingCanvas={() => this.onNavigationClick('imageMasker')} />);
      case 'textAnotator': return (<div>textAnotator</div>);
      case 'imageMasker': return (<MaskingCanvas />);
      default: return (<div>home</div>);
    }
  }

  render() {
    return (
      <div className="App">
        <NavigationBar onClick={this.onNavigationClick} />
        <div className="App-body-card">
          {this.renderBody()}
        </div>
      </div>
    );
  }
}

export default App;
