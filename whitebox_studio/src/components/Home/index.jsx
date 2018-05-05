import React from 'react';
import './Home.css';

class ExperimentCard extends React.Component {
  render() {
    const {
      name, description, domain, started,
    } = this.props.experiment;
    return (
      <div className="Home-ExperimentCard">
        <div className="Home-ExperimentCard-info">
          <h1>{name}</h1>
          <p>{description}</p>
          <h2>{domain ? 'image' : 'text'}</h2>
        </div>
        <button className="Home-ExperimentCard-button" onClick={this.props.goToWorkspace}>{started ? 'resume' : 'start'}</button>
      </div>
    );
  }
}

class Home extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      all_experiments: [],
    };
  }

  componentDidMount = () => {
    // TODO get running experiments
    this.setState({
      all_experiments: [
        {
          name: 'Decoy MNIST',
          description: 'Handwritten digits',
          domain: 1,
          started: true,
        },
        {
          name: 'Newsgroup-20',
          description: 'Emails',
          domain: 0,
          started: false,
        },
      ],
    });
  }

  goToWorkspace = (experiment) => () => {this.props.goToWorkspace(experiment);}

  renderBody = () => this.state.all_experiments.map(experiment => (<ExperimentCard experiment={experiment} goToWorkspace={this.goToWorkspace(experiment)}/>))

  render() {
    return (
      <div className="Home">
        {this.renderBody()}
      </div>
    );
  }
}

export default Home;
