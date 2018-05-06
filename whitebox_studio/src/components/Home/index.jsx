import React from 'react';
import './Home.css';


// const API = 'http://whitebox-rfft.herokuapp.com/api/v1';
const API = `http://55c9e8ce.ngrok.io/api/v1`;
// const API = 'http://localhost:8000/api/v1';


class ExperimentCard extends React.Component {
  getDomainName = (domainId) => {
    switch(domainId) {
      case 0:
        return 'Text';
      case 1:
        return 'Image';
      case 2:
        return 'Tabular';
      default:
        return 'Other';
    }
  }

  render() {
    const {
      name, description, domain, started,
    } = this.props.experiment;
    return (
      <div className="Home-ExperimentCard">
        <div className="Home-ExperimentCard-info">
          <h1 className="Home-ExperimentCard-name">{name}</h1>
          <p className="Home-ExperimentCard-desc">{description}</p>
          <h2 className="Home-ExperimentCard-domain">{this.getDomainName(domain)}</h2>
        </div>
        <button className="Home-ExperimentCard-button" onClick={this.props.goToWorkspace}>{started ? 'Resume experiment' : 'Start experiment'}</button>
      </div>
    );
  }
}

class ExperimentList extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      all_experiments: [],
    };
  }

  componentDidMount = () => {
    fetch(API + '/all_experiments')
      .then(response => response.json())
      .then(data => this.setState({ all_experiments: data.all_experiments }));
  }

  goToWorkspace = (experiment) => () => {
    this.props.goToWorkspace(experiment);
    fetch(API + `/experiment/DecoyMNIST`, {method: 'POST'})
      .then(response => response.json())
      .then(data => this.setState({ all_experiments: data.all_experiments }));
  }

  render() {
    return this.state.all_experiments.map(experiment => (
      <ExperimentCard experiment={experiment} goToWorkspace={this.goToWorkspace(experiment)}/>
    ))
  }
}

class Home extends React.Component {
  render() {
    return (
      <div className="Home">
        <ExperimentList goToWorkspace={this.props.goToWorkspace}/>
      </div>
    );
  }
}

export default Home;
