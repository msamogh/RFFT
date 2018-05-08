import React from 'react';
import constants from '../../constants';
import './Home.css';
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


class PreTrainedExperimentCard extends React.Component {
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
      name, description, domain, epochs, annotations, annotationWeights, testAccuracy, trainingAccuracy,
    } = this.props.experiment;
    return (
      <div className="Home-ExperimentCard">
        <div className="Home-ExperimentCard-info">
          <h1 className="Home-ExperimentCard-name">{name}</h1>
          <p className="Home-ExperimentCard-desc">{description}</p>

          <div className="Home-ExperimentCard-params">
            <p className="Home-ExperimentCard-desc">{`epochs ${epochs}`}</p>
            <p className="Home-ExperimentCard-desc">{`annotations ${annotations}`}</p>
            <p className="Home-ExperimentCard-desc">{`annotationWeights ${annotationWeights}`}</p>
            <p className="Home-ExperimentCard-desc">{`testAccuracy ${testAccuracy}%`}</p>
            <p className="Home-ExperimentCard-desc">{`trainingAccuracy ${trainingAccuracy}%`}</p>
          </div>

          <h2 className="Home-ExperimentCard-domain">{this.getDomainName(domain)}</h2>
        </div>
        <button className="Home-ExperimentCard-button" onClick={this.props.goToExplain}>Explain</button>
      </div>
    );
  }
}
class ExperimentList extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      all_experiments: [],
      pre_trained_expereiments: [],
    };
  }

  componentDidMount = () => {
  //   const all_experiments =[
  //       {
  //         "description": "Handwritten digits",
  //         "domain": 1,
  //         "id": "DecoyMNIST",
  //         "initialized": true,
  //         "name": "Decoy MNIST"
  //       },
  //       {
  //         "name": "Newsgroup-20",
  //         "description": "Emails",
  //         "domain": 0,
  //         "started": false
  //       }
  //   ]
  //   const pre_trained_expereiments =[
  //     {
  //       "description": "Handwritten digits",
  //       "domain": 1,
  //       "id": "DecoyMNIST",
  //       "initialized": true,
  //       "name": "Decoy MNIST",
  //       epochs: 4,
  //       annotations: 5,
  //       annotationWeights: 5,
  //       testAccuracy: 95,
  //       trainingAccuracy: 96,
  //     },
  //     {
  //       "description": "Handwritten digits",
  //       "domain": 1,
  //       "id": "DecoyMNIST",
  //       "initialized": true,
  //       "name": "Decoy MNIST",
  //       epochs: 4,
  //       annotations: 5,
  //       annotationWeights: 5,
  //       testAccuracy: 95,
  //       trainingAccuracy: 96,
  //     },
  //     {
  //       "name": "Newsgroup-20",
  //       "description": "Emails",
  //       "domain": 0,
  //       "started": false,
  //       epochs: 4,
  //       annotations: 5,
  //       annotationWeights: 5,
  //       testAccuracy: 95,
  //       trainingAccuracy: 96,
  //     }
  // ]
  //   this.setState({all_experiments, pre_trained_expereiments})
    const API = `${constants.API}/all_experiments`;
    fetch(API)
      .then(response => response.json())
      .then(data => this.setState({ all_experiments: data.all_experiments }));
  }

  goToWorkspace = (experiment) => () => {
    this.props.goToWorkspace(experiment);
  }

  goToExplain = (experiment) => () => {
    this.props.goToExplain(experiment);
  }

  renderCards = () => this.state.all_experiments.map(experiment => (
    <ExperimentCard key={experiment.id} experiment={experiment} goToWorkspace={this.goToWorkspace(experiment)}/>
  ))

  renderPreCards = () => this.state.pre_trained_expereiments.map(experiment => (
    <PreTrainedExperimentCard key={experiment.id} experiment={experiment} goToExplain={this.goToExplain(experiment)}/>
  ))

  render() {
    return (
      <div lassName="Home">
      <div className="experimentList">
        {this.renderCards()}
      </div>
      <div className="experimentList">
        {this.renderPreCards()}
      </div>
      </div>
    );
  }
}

class Home extends React.Component {
  render() {
    return (
      <div className="Home">
        <ExperimentList goToWorkspace={this.props.goToWorkspace} goToExplain={this.props.goToExplain}/>
      </div>
    );
  }
}

export default Home;
