from faker import Faker

from enviroment.environment_config import GreekMythProvider, Configuration


def create_config(alpha, batch_size, fc1, fc2):

    fake = Faker()
    fake.add_provider(GreekMythProvider)  # Register the provider first

    network_name = fake.greek_name()  # Then use the custom method
    return Configuration(alpha=alpha, beta=alpha * 2, tau=alpha * 5,
                         fc1=fc1, fc2=fc2, batch_size=batch_size,
                         noise=NOISE, gamma=GAMMA, load_checkpoint=False, steps=STEPS, unique_name=network_name)
