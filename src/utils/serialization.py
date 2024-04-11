import pickle
import aiofiles
import os


class Serializer:
    @staticmethod
    async def serialize_interactions_to_disk(interactions, directory_to_serialize_into):
        # TODO : Make proper async
        for interaction in interactions:
            file_path = os.path.join(directory_to_serialize_into, interaction.interaction_id) + ".pkl"
            await Serializer.async_serialize_to_disk(interaction, file_path)

    @staticmethod
    async def deserialize_interactions_from_disk(directory_to_deserialize_from):
        #TODO : Make proper async
        interactions = []
        for file in os.listdir(directory_to_deserialize_from):
            if file.endswith(".pkl"):
                file_path = os.path.join(directory_to_deserialize_from, file)
                interaction = await Serializer.async_deserialize_from_disk(file_path)
                interactions.append(interaction)
        return interactions

    @staticmethod
    async def serialize_pulse_document_to_disk(pulse_document, directory_to_serialize_into):
        # TODO : Make proper async
        file_path = os.path.join(directory_to_serialize_into, str(pulse_document.document_id)) + ".pkl"
        await Serializer.async_serialize_to_disk(pulse_document, file_path)

    @staticmethod
    async def deserialize_pulse_documents_from_disk(directory_to_deserialize_from):
        # TODO : Make proper async
        pulse_documents = []
        for file in os.listdir(directory_to_deserialize_from):
            if file.endswith(".pkl"):
                file_path = os.path.join(directory_to_deserialize_from, file)
                pulse_document = await Serializer.async_deserialize_from_disk(file_path)
                pulse_documents.append(pulse_document)
        return pulse_documents

    @staticmethod
    def serialize_to_disk(obj, filename):
        serialized_data = pickle.dumps(obj)
        with open(filename, 'wb') as f:
            f.write(serialized_data)

    @staticmethod
    def deserialize_from_disk(filename):
        with open(filename, 'rb') as f:
            serialized_data = f.read()
        return pickle.loads(serialized_data)

    @staticmethod
    async def async_serialize_to_disk(obj, filename):
        serialized_data = pickle.dumps(obj)
        async with aiofiles.open(filename, 'wb') as f:
            await f.write(serialized_data)

    @staticmethod
    async def async_deserialize_from_disk(filename):
        async with aiofiles.open(filename, 'rb') as f:
            serialized_data = await f.read()
        return pickle.loads(serialized_data)