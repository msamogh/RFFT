import React from 'react';
import NavigationBar from '../NavigationBar';
import MaskingCanvas from '../MaskingCanvas';
import './App.css';

class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      page: 'home'
    };
  }

  onNavigationClick = (page) => {
    this.setState({page});
  }

  renderBody = () => {
    switch (this.state.page) {
      case 'home': return (<div>home</div>);
      case 'textAnotator': return (<div>textAnotator</div>);
      case 'imageMasker': return (<MaskingCanvas/>);
      default: return (<div>home</div>);
    }
  }

  render() {
    return (
      <div className="App">
        <NavigationBar onClick={this.onNavigationClick}/>
        {this.renderBody()}
      </div>
    );
  }
}

export default App;
