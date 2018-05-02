import React from 'react';
import MaskingCanvas from '../MaskingCanvas';
import './Home.css';

class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {

      page: 1,
    };
  }

  onOptionClick = (option, value) => () => {
    const { page } = this.state;
    this.setState({ [option]: value, page: page + 1 });
    if (page === 2) {
      this.props.navigateToMaskingCanvas();
    }
  }

  renderBody = () => {
    switch (this.state.page) {
      case 1: return (
        <div>
          <h1>Are you a </h1>
          <button onClick={this.onOptionClick('experience', 'newbie')}>newbie</button>
          <button onClick={this.onOptionClick('experience', 'data scientist')}>data scientist</button>
        </div>
      );
      case 2: return (
        <div>
          <h1>Choose a dataset</h1>
          <h2>text</h2>
          <button onClick={this.onOptionClick('dataset', 'newsgroup')}>newsgroup</button>
          <h2>image</h2>
          <button onClick={this.onOptionClick('dataset', 'decoy mnist')}>decoy mnist</button>
          <button onClick={this.onOptionClick('dataset', 'CIFAR')}>CIFAR</button>

        </div>
      );
      case 3: return (<MaskingCanvas />);
      default: return (<div>home</div>);
    }
  }

  render() {
    return (
      <div className="Home">
        <div className="App-body-card">
          {this.renderBody()}
        </div>
      </div>
    );
  }
}

export default App;
