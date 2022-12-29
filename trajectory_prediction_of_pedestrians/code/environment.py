import yaml

class ApplicationProperties( object ):

    def __init__( self, propertiesFilePath: str ) -> None:
        self._propertiesFile = propertiesFilePath
        self._applicationProperties = None

    def initializeProperties( self ) -> dict:
        with open( self._propertiesFile, 'r') as file:
            self._applicationProperties = yaml.safe_load(file)

    def get_property_value( self, propertyName: str ):
        properties = propertyName.split( "." )

        propertyTree = self._applicationProperties
        for p in properties:
            if p not in propertyTree:
                 return None
            else:
                propertyTree = propertyTree[ p ]

        return propertyTree